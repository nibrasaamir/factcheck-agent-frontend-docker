import os
import warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import argparse
import json
from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from tools.web_search import web_search_tool
from tools.document_fetcher import document_fetcher_tool
from tools.summarizer import summarizer_tool
from graph.graph_manager import GraphManager

load_dotenv()

def build_agent():
    # OpenAI LLM
    llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # my tools
    tools = [web_search_tool, document_fetcher_tool, summarizer_tool]

    # zero-shot REACT agent - AGENTEXECUTOR
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True,
        agent_kwargs={
            "prefix": (
                "You are an autonomous fact-checking agent.  \n"
                "For each claim, follow exactly these steps in order:\n\n"
                "Call 'web_search' with the exact claim text that user provided in the input action.\n"
                "CALL `web_search` with the exact claim text to get up to 4 URLs.\n\n"
                "FOR EACH URL, in the order returned:\n"
                "  CALL `document_fetcher` on that single URL.  \n"
                "  IF the returned text is empty or shorter than 100 characters, **skip** that URL and move on to the next one.  \n"
                "  OTHERWISE, IMMEDIATELY call `summarize_and_classify` on that fetched text.  \n"
                " **Do not** summarize any empty/error text or any metadata.\n\n"
                "REPEAT step 2 until you have **3** successfully fetched+summarized articles.\n\n"
                "**Then** stop calling tools and **output** your final answer and verdict.  \n"
                "   – Do **not** batch multiple articles into one `summarize_and_classify` call.\n"
            )
        }
    )
    return agent


def extract_urls(raw_text, max_urls = 4):

    # uses serpAPI to find urls
   
    lines = [ln.strip() for ln in raw_text.splitlines()]
    urls = [u for u in lines if u.startswith("http")]
    return urls[:max_urls]

#-------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autonomous Fact-Checking Agent")

    parser.add_argument( "claims", nargs="+", help='One or more claims, e.g. "Area 51 has active UFO research"')

    args = parser.parse_args()
    agent = build_agent()
    gm = GraphManager()

    for claim in args.claims:
        print("\n Claim: ", claim)
        claim_id = gm.add_claim(claim)

        # Run the agent: returns final answer + tool steps

        result = agent({"input": claim})
        final_answer = result["output"]
        steps = result["intermediate_steps"]

        # keeping a list to track agent's findings.
        support_scores, refute_scores = [], []
        
        # each step is provided by my agent. Iterating over it.
        for action, observation in steps:
            tool = action.tool
            inp  = action.tool_input
            obs  = observation

            if tool == "web_search":
                urls = extract_urls(obs)
                print(f"Agent found {len(urls)} URLs:")
                for u in urls:
                    print("  ", u)

            elif tool == "document_fetcher":
                url  = inp
                text = obs
                if not text.strip():
                    print(f"  → skipped {url} (empty or error)")
                    continue
                doc_id = gm.add_document(url, text)
                gm.add_edge(claim_id, doc_id, "cites")
                print(f"Fetched & ingested document: {url}")

            elif tool == "summarize_and_classify":
                data = json.loads(obs)
                snippet_id = gm.add_snippet(
                    summary=data["summary"],
                    label=data["label"],
                    score=data["score"]
                )
                rel = data["label"]
                if rel in ("supports", "refutes"):
                    gm.add_edge(claim_id, snippet_id, rel)
                    if rel == "supports":
                        support_scores.append(data["score"])
                    else:
                        refute_scores.append(data["score"])
                    print(f"  → snippet {rel} ({data['score']:.2f})")
                else:
                    print("  → snippet neutral; no edge added")

        # Verdict:
        avg_sup = sum(support_scores)/len(support_scores) if support_scores else 0.0
        avg_ref = sum(refute_scores)/len(refute_scores) if refute_scores else 0.0
        if avg_sup > avg_ref:
            verdict, confidence = "Supported", avg_sup
        elif avg_ref > avg_sup:
            verdict, confidence = "Refuted", avg_ref
        else:
            verdict, confidence = "Inconclusive", max(avg_sup, avg_ref)

        # Final Answer:
        print(f"\n>>> Agent’s Final Answer:\n{final_answer}\n")
        print(f">>> Verdict: {verdict} (confidence: {confidence:.2f})")
        print("\nGraph stats:", gm.stats())

        print("\nTop supporting snippets:")
        for summ, sc in gm.get_top_snippets(claim_id, "supports"):
            print(f" • ({sc:.2f}) {summ}")
        print("\nTop refuting snippets:")
        for summ, sc in gm.get_top_snippets(claim_id, "refutes"):
            print(f" • ({sc:.2f}) {summ}")

        # More options for the user: 
        print("\nYou can now query the graph for this claim.")
        print("Commands:")
        print("  docs     → list all document URLs ingested")
        print("  supports → show all supporting snippets")
        print("  refutes  → show all refuting snippets")
        print("  exit     → move on to next claim")

        while True:
            cmd = input("\n> ").strip().lower()
            if cmd in ("exit", "quit"):
                break

            elif cmd in ("docs", "documents"):
                doc_ids = [
                    tgt for src, tgt, rel in gm.edges
                    if src == claim_id and rel == "cites"
                ]
                if not doc_ids:
                    print("  (no documents ingested)")
                else:
                    for did in doc_ids:
                        url = gm.nodes[did]["properties"]["url"]
                        print(f"  • {url}")

            elif cmd in ("supports", "support"):
                snippets = gm.get_top_snippets(claim_id, "supports", k=len(gm.nodes))
                if not snippets:
                    print("  (no supporting snippets)")
                else:
                    for summ, sc in snippets:
                        print(f"  • ({sc:.2f}) {summ}")

            elif cmd in ("refutes", "refute"):
                snippets = gm.get_top_snippets(claim_id, "refutes", k=len(gm.nodes))
                if not snippets:
                    print("  (no refuting snippets)")
                else:
                    for summ, sc in snippets:
                        print(f"  • ({sc:.2f}) {summ}")

            else:
                print("Unknown command. Valid commands: docs, supports, refutes, exit")

if __name__ == "__main__":
    main()


