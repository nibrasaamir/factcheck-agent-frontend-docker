import os
import re
import warnings
import json
import streamlit as st
from transformers import logging as hf_logging

# suppress transformer logs
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

from verify import build_agent, extract_urls
from graph.graph_manager import GraphManager

st.set_page_config(page_title="Agentic Fact-Checker", layout="wide")
st.title("🕵️ Autonomous Fact-Checking Agent")

st.markdown(
    """
Enter one or more claims separated by commas or enclosed in quotes.  
Examples:  
- `"The Eiffel Tower is painted every 7 years", Area 51 has active UFO research`  
- `Covid vaccines cause microchips, 2+2=4`  
"""
)

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = {}

def parse_claims(text: str) -> list[str]:
    quotes = re.findall(r'"([^"]+)"', text)
    if quotes:
        return [q.strip() for q in quotes if q.strip()]
    return [seg.strip() for seg in text.split(",") if seg.strip()]

# --- Input Form ---
with st.form("verify_form", clear_on_submit=False):
    user_input = st.text_area("Your claims here:", height=100)
    submitted = st.form_submit_button("🔍 Verify Claims")
    if submitted and user_input.strip():
        claims = parse_claims(user_input)
        agent = build_agent()
        st.session_state.results = {}  # reset previous results

        for claim in claims:
            gm = GraphManager()
            logs: list[str] = []
            support_scores, refute_scores = [], []

            # Run agent
            result = agent({"input": claim})
            final_answer = result["output"]
            steps = result["intermediate_steps"]

            logs.append("=== Agent Trace ===")
            for action, observation in steps:
                tool = action.tool
                inp = action.tool_input
                obs = observation or ""
                logs.append(f"→ Tool: {tool}")
                logs.append(f"   Input: {inp}")
                logs.append(f"   Obs:   {obs}")

                if tool == "web_search":
                    urls = extract_urls(obs)
                    logs.append(f"   Parsed URLs ({len(urls)}):")
                    for u in urls:
                        logs.append(f"     • {u}")

                elif tool == "document_fetcher":
                    url = inp
                    if obs.strip():
                        doc_id = gm.add_document(url, obs)
                        gm.add_edge(claim, doc_id, "cites")
                        logs.append(f"   Ingested document: {url}")
                    else:
                        logs.append(f"   Skipped empty fetch for {url}")

                elif tool == "summarize_and_classify":
                    data = json.loads(obs)
                    snippet_id = gm.add_snippet(
                        summary=data["summary"],
                        label=data["label"],
                        score=data["score"]
                    )
                    rel = data["label"]
                    if rel in ("supports", "refutes"):
                        gm.add_edge(claim, snippet_id, rel)
                        if rel == "supports":
                            support_scores.append(data["score"])
                        else:
                            refute_scores.append(data["score"])
                        logs.append(f"   → snippet {rel} ({data['score']:.2f})")
                    else:
                        logs.append("   → snippet neutral")

            # Verdict
            avg_sup = sum(support_scores)/len(support_scores) if support_scores else 0.0
            avg_ref = sum(refute_scores)/len(refute_scores) if refute_scores else 0.0
            if avg_sup > avg_ref:
                verdict, conf = "Supported", avg_sup
            elif avg_ref > avg_sup:
                verdict, conf = "Refuted", avg_ref
            else:
                verdict, conf = "Inconclusive", max(avg_sup, avg_ref)

            # Save results in session_state
            st.session_state.results[claim] = {
                "final_answer": final_answer,
                "verdict": verdict,
                "confidence": conf,
                "graph_manager": gm,
                "logs": logs
            }

# --- Display Results ---
if st.session_state.results:
    for claim, data in st.session_state.results.items():
        gm: GraphManager = data["graph_manager"]

        st.markdown("---")
        st.header(f"Claim: {claim}")

        # Colored headings
        st.markdown(f"<h3 style='color:blue'>Final Answer:</h3>", unsafe_allow_html=True)
        st.write(data["final_answer"])

        st.markdown(f"<h3 style='color:green'>Verdict:</h3> {data['verdict']} (confidence {data['confidence']:.2f})", unsafe_allow_html=True)
        st.markdown(f"<h4>Graph stats:</h4> {gm.stats()}", unsafe_allow_html=True)

        st.subheader("Top Supporting Snippets")
        for summ, sc in gm.get_top_snippets(claim, "supports"):
            st.markdown(f"<span style='color:darkgreen'>• ({sc:.2f}) {summ}</span>", unsafe_allow_html=True)

        st.subheader("Top Refuting Snippets")
        for summ, sc in gm.get_top_snippets(claim, "refutes"):
            st.markdown(f"<span style='color:darkred'>• ({sc:.2f}) {summ}</span>", unsafe_allow_html=True)

        # Interactive graph commands
        st.markdown("**Query the knowledge graph:**")
        cmd = st.selectbox(
            "Select a command",
            ("docs", "supports", "refutes", "exit"),
            key=f"cmd_{claim}"
        )
        if cmd in ("docs", "documents"):
            doc_ids = [tgt for src, tgt, rel in gm.edges if src == claim and rel == "cites"]
            if not doc_ids:
                st.info("No documents ingested.")
            else:
                for did in doc_ids:
                    st.write(f"• {gm.nodes[did]['properties']['url']}")

        elif cmd in ("supports", "support"):
            snippets = gm.get_top_snippets(claim, "supports", k=len(gm.nodes))
            if not snippets:
                st.info("No supporting snippets.")
            else:
                for summ, sc in snippets:
                    st.write(f"• ({sc:.2f}) {summ}")

        elif cmd in ("refutes", "refute"):
            snippets = gm.get_top_snippets(claim, "refutes", k=len(gm.nodes))
            if not snippets:
                st.info("No refuting snippets.")
            else:
                for summ, sc in snippets:
                    st.write(f"• ({sc:.2f}) {summ}")

        # Logs in an expander
        st.markdown("**Agent Processing Log**")
        with st.expander("Show agent trace and logs"):
            for line in data["logs"]:
                st.text(line)
