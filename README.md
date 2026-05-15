# SHL Assessment Recommendation Agent

A conversational AI agent that helps hiring managers find the right SHL assessments through natural multi-turn dialogue. Built with FastAPI, BM25 retrieval, and Groq-hosted LLaMA 3.

---

## Features

- Asks clarifying questions when hiring intent is vague
- Recommends 1–10 SHL Individual Test assessments grounded in the catalog
- Refines shortlists mid-conversation when constraints change
- Compares assessments side-by-side using catalog evidence only
- Refuses off-topic requests and prompt injection attempts
- Stateless API — full conversation history sent per request

---

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── agent.py          # Core LLM orchestration and response logic
│   ├── catalog_loader.py # Loads SHL catalog JSON from disk
│   ├── main.py           # FastAPI, /health and /chat endpoints
│   ├── models.py         # Request/response schemas
│   ├── prompts.py        # System prompt with conversation behavior rules
│   ├── retriever.py      # BM25 hybrid retriever with domain boosting
│   └── utils.py          # Vagueness and scope violation detection
├── data/
│   └── shl_product_catalog.json   
├── evaluate.py          
├── requirements.txt
├── .env
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/shl-agent.git
cd shl-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

DEPLOYED ON RENDER: https://shl-recommender-7r0b.onrender.com/docs
TRY IT OUT: Click Chat then try it out and then execute.

