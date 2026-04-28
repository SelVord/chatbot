# RAG Chatbot Builder

A local web application that lets you build, train, and export custom AI chatbots powered by your own documents. Upload PDFs, Word files, or paste text, and the chatbot answers questions strictly based on that knowledge. When you're done, export the bot as a self-contained package and embed it on any webpage.

---

## Features

- **Multi-session management** — create and switch between independent chatbot sessions, each with its own knowledge base and settings
- **Document indexing** — upload PDF, DOCX, or TXT files; the content is chunked, embedded, and stored in a FAISS vector index
- **Text paste & edit** — paste raw text directly, and edit it later in-place; changes are re-indexed automatically
- **Retrieval-Augmented Generation (RAG)** — every answer is grounded in your documents; the bot says "I don't have information about that" when something is outside its knowledge
- **Natural conversation** — greetings, identity questions, and off-topic questions are handled gracefully without generic error messages
- **Streaming responses** — answers appear token by token as the LLM generates them
- **Configurable LLM** — switch between OpenAI (GPT-4o, GPT-4o-mini, etc.) and local Ollama models; available models are auto-detected
- **Configurable embeddings** — OpenAI `text-embedding-3-small` or a free local model (`all-MiniLM-L6-v2`) that runs fully on CPU
- **Per-session bot persona** — set a custom name and system prompt for each session
- **Relevance threshold** — tune how strictly the bot filters retrieved chunks before answering
- **Export** — download a ZIP containing a ready-to-deploy FastAPI server and a copy-paste JavaScript chat widget for any webpage
- **Chat history export** — download any session's conversation as JSON
- **Dark UI** — clean dark interface built with Streamlit

---

## Architecture

```
chatbot/
├── app.py                   Main Streamlit application
├── config.py                Central configuration (paths, defaults, env vars)
├── requirements.txt         Python dependencies
├── .env                     Environment variables (API keys, model settings)
├── .streamlit/
│   └── config.toml          Streamlit server settings
└── data/
    ├── chat_history.db      SQLite database (sessions, messages, documents)
    ├── uploads/             Uploaded and pasted files stored on disk
    ├── vector_stores/       FAISS indices, one directory per session
    └── core/
        ├── database.py      SQLite operations (sessions, messages, documents)
        ├── document_processor.py  File loading and text chunking
        ├── embeddings.py    Embedding model factory (OpenAI / local)
        ├── export.py        Export ZIP builder (server + widget templates)
        ├── llm.py           LLM factory (OpenAI / Ollama)
        ├── rag_chain.py     RAG pipeline — retrieval, prompting, streaming
        └── vector_store.py  FAISS operations (build, load, update, delete)
```

### How a Question is Answered

1. The question is embedded using the same model that was used during indexing
2. FAISS returns the top-5 most similar document chunks with relevance scores
3. Chunks below the relevance threshold are filtered out; if nothing passes, the top-1 result is used as a fallback
4. The retrieved chunks are formatted as context and sent to the LLM alongside the system prompt and recent conversation history
5. The LLM streams its response token by token
6. Source cards (filename, page, score, snippet) are shown below the answer

---

## Requirements

- Python 3.10 or higher
- [Ollama](https://ollama.com) — only if you want to use local models
- An OpenAI API key — only if you want to use OpenAI models or embeddings

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd chatbot

# 2. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Edit the `.env` file in the project root:

```ini
# LLM provider: "openai" or "ollama"
LLM_PROVIDER=ollama

# OpenAI settings (required when LLM_PROVIDER=openai)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Ollama settings (required when LLM_PROVIDER=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Embedding provider: "openai" or "local"
EMBEDDING_PROVIDER=local

# Local embedding model (used when EMBEDDING_PROVIDER=local)
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Choosing a provider

| Scenario | LLM_PROVIDER | EMBEDDING_PROVIDER | Notes |
|---|---|---|---|
| Fully local, free | `ollama` | `local` | Requires Ollama. No API key needed. |
| OpenAI, cloud | `openai` | `openai` | Requires `OPENAI_API_KEY`. Best quality. |
| Mixed | `openai` | `local` | Local embeddings, OpenAI for generation. |

---

## Running

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

If you use Ollama, make sure it is running first:

```bash
ollama serve
ollama pull llama3.2   # or whichever model you want
```

---

## Usage Guide

### 1. Create a Session

In the left sidebar, type a name for your chatbot session and click **➕**. Each session has its own knowledge base, settings, and chat history. The app starts with no session selected — nothing loads until you create or choose one.

### 2. Configure the Bot

After creating a session, the **⚙️ Bot Settings** section appears:

- **Bot name** — the name the bot uses to introduce itself
- **System prompt** — instructions defining the bot's role, tone, and domain
- **Model settings** (inside the expander):
  - **LLM provider** — `openai` or `ollama`
  - **Model** — dropdown; for Ollama, all installed models are auto-detected
  - **Embeddings** — `openai` or `local` (local runs on CPU, no API key needed)
  - **Relevance threshold** — chunks below this score are excluded from context (default 0.20)

Click **💾 Save settings** to apply.

> **Important:** if you change the embedding provider after indexing, delete the session and re-index — the new embeddings are incompatible with the old FAISS index.

### 3. Add Knowledge

Under **📄 Knowledge Base**, toggle between two modes:

**📁 Upload file**
Select one or more PDF, DOCX, or TXT files and click **📥 Index files**. Each file is split into overlapping 1000-character chunks, embedded, and stored in the FAISS vector index.

**✏️ Paste text**
Give the text a label (e.g. "Company FAQ"), paste the content, and click **📥 Index text**. The text is saved as a `.txt` file so it can be edited later. Click the **✏️** button next to any `.txt` document in the knowledge base list to open an inline editor. Saving re-indexes the session automatically.

### 4. Chat

Once at least one document is indexed, the chat input appears. Type your question and press Enter.

The bot:
- Answers greetings and questions about itself naturally from its configured persona
- Answers knowledge questions only from indexed content
- Responds specifically when a topic is outside its knowledge, e.g. *"I don't have information about your food preferences"*
- Streams the response token by token
- Shows **📚 Sources** below each answer (filename, page number, relevance score, text snippet)

### 5. Export

Under **📦 Export Bot**, click **⬇️ Download export package (ZIP)** to get a deployable package containing your bot's server, knowledge base, and a chat widget. See [Exporting](#exporting) below.

### 6. Session Management

The **🗑️ Danger zone** expander contains:
- **Clear chat history** — removes all messages, keeps the knowledge base
- **Delete this session** — removes the session, index, and all messages permanently
- **Export chat (JSON)** — downloads the full conversation history

---

## Exporting

The export package lets you host your chatbot on any server and embed it on any webpage with a single copy-paste.

### What's in the ZIP

| File | Description |
|---|---|
| `vector_store/` | FAISS index files (the trained knowledge base) |
| `config.json` | Bot name, system prompt, model and embedding settings |
| `server.py` | Standalone FastAPI server with a streaming `/chat` endpoint |
| `widget.html` | Self-contained chat widget demo with an embeddable snippet |
| `requirements.txt` | Python dependencies for the server |
| `.env.example` | Environment variable template |
| `README.md` | Full deployment instructions |

### Deploying the Server

```bash
pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY if needed
uvicorn server:app --host 0.0.0.0 --port 8000
```

**API endpoints:**
- `POST /chat` — body: `{ "message": "...", "history": [...] }` → streaming plain-text response
- `GET /health` — returns `{ "status": "ok", "bot": "..." }`

### Embedding the Widget on Your Webpage

Open `widget.html` and copy everything between `<!-- WIDGET START -->` and `<!-- WIDGET END -->`. Paste it into your webpage's `<body>`. Then update the `API` constant to your server's public URL:

```js
var API = "https://your-server.com/chat";
```

The widget renders as a floating chat bubble in the bottom-right corner. It streams responses and keeps conversation history within the page session.

### Hosting Options

| Platform | Notes |
|---|---|
| [Render](https://render.com) | Free tier, GitHub deploy in minutes |
| [Railway](https://railway.app) | Free starter plan, zero config |
| [Fly.io](https://fly.io) | Generous free tier, global edge |
| VPS / VM | Full control; run `uvicorn` as a systemd service |

A Dockerfile template is included in the exported `README.md`.

> **LLM provider after export:**
> - **OpenAI** — works on any server; set `OPENAI_API_KEY` in `.env`
> - **Ollama** — requires Ollama installed on the deployment server; pull the model first: `ollama pull <model>`

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'core'`**
Run the app from the project root: `cd chatbot && streamlit run app.py`

**Ollama 404 — model not found**
Pull the model first: `ollama pull llama3.2`
Then open Model settings, select the correct model from the dropdown, and save.

**Bot always says it doesn't have information**
- Lower the Relevance threshold slider in Model settings (try 0.10)
- Make sure documents were indexed under the current session
- If the embedding provider was changed after indexing, delete the session and re-index

**Slow first response with local embeddings**
The `all-MiniLM-L6-v2` model (~90 MB) is downloaded on first use. Subsequent runs use the cached model.

**`Failed to fetch dynamically imported module` in browser**
Hard-refresh the page (`Ctrl+Shift+R` on Windows/Linux, `Cmd+Shift+R` on macOS) to clear cached JavaScript.

**OpenAI API errors**
Verify `OPENAI_API_KEY` is correct in `.env` and your account has available credits.

---

## Tech Stack

| Package | Role |
|---|---|
| `streamlit` | Web UI |
| `langchain` / `langchain-community` | RAG orchestration, document loaders |
| `langchain-openai` | OpenAI LLM and embeddings |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` + `torch` | Local embedding model |
| `pypdf` | PDF parsing |
| `docx2txt` | DOCX parsing |
| `openai` | OpenAI API client |
| `python-dotenv` | `.env` loading |
| `fastapi` + `uvicorn` | Exported chatbot server |
