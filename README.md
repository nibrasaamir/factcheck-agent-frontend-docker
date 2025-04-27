# Autonomous Fact-Checking Agent

An end-to-end, autonomous "verification assistant" that plans and executes fact–checking for arbitrary claims.  It uses a LangChain Agent to:

1. **Search** a news API for relevant articles  
2. **Fetch & parse** the article text  
3. **Summarize & classify** each snippet as supporting or refuting  
4. **Ingest** everything into an in-memory knowledge graph  
5. **Report** a verdict (Supported / Refuted / Inconclusive) with confidence and top evidence  

A Streamlit frontend lets users enter one or more claims, see results, explore the knowledge graph, and inspect the agent’s full tool trace.

---

## 🚀 Installation

### Prerequisites

- **Python 3.10**  
- [Pipenv](https://pipenv.pypa.io/en/latest/)  
- API keys in a `.env` file at project root:
  ```bash
  OPENAI_API_KEY=your_openai_key
  SERPAPI_API_KEY=your_serpapi_key
  GROQ_API_KEY=your_groq_key
  ```

### Local Setup

```bash
git clone https://github.com/your-org/factcheck-agent.git
cd factcheck-agent
pip install pipenv
pipenv install --deploy --system --ignore-pipfile
pipenv run python verify.py "The S&P stock rose 0.74%"
pipenv run streamlit run app.py
```

### With Docker

```bash
docker build -t factcheck-agent-frontend-docker -f Dockerfile .      

docker run --rm \  --env-file .env \  -v ~/.cache/huggingface:/app/cache/huggingface \
  -p 8501:8501 \
  factcheck-agent-frontend-docker

```

---

## 💻 Usage Examples

### CLI

```bash
pipenv run python verify.py "Area 51 has active UFO research"
pipenv run streamlit run app.py
```

### Streamlit Frontend

```bash
pipenv run streamlit run app.py
```
Open http://localhost:8501.



## 🛠️ Challenges

- Noisy page scrapes → newspaper3k + BeautifulSoup fallback  
- Empty fetches → skip until 3 valid sources  
- Token limits → chunk & re-summarize  
- Model loading in Docker → HF cache dir, Groq fallback  
- Streamlit state resets → session_state  

---

## 🔮 Next Steps

- GUI enhancements: thought bubbles, network graph  
- Real-time monitoring: scheduled re-checks  
- Rebuttal generation: auto-draft summaries  
- Persistence: switch to Neo4j/SQLite backend  

---

*Happy fact-checking!*
