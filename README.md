# RAG Chatbot Builder

A local web application that lets you build, train, and export custom AI chatbots powered by your own documents. Upload PDFs, Word files, or paste text — the chatbot answers questions strictly based on that knowledge. When you're done, export the bot as a self-contained package and embed it on any webpage.

---

## Quick Start

> **Choose your path before you begin:**
> - **Free / local** — uses [Ollama](https://ollama.com) + local embeddings. No API key needed.
> - **Cloud / best quality** — uses OpenAI. Requires an API key from [platform.openai.com](https://platform.openai.com).

### 1. Install Python dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment

Open the `.env` file in the project root and set your provider:

**Option A — Ollama (free, local)**
```ini
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
EMBEDDING_PROVIDER=local
```

**Option B — OpenAI (cloud)**
```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
```

### 3. Start Ollama (only if using Ollama)

```bash
# In a separate terminal — keep it running
ollama serve

# Pull the model you want (first time only)
ollama pull llama3.2
```

### 4. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 5. Create your first chatbot

1. Type a name in the sidebar and click **➕** to create a session
2. Upload a file or paste text under **📄 Knowledge Base**
3. Click **Index files** or **Index text**
4. Start chatting in the main area

---

## Features

- **Multi-session management** — independent sessions each with their own knowledge base, settings, and chat history
- **Document indexing** — upload PDF, DOCX, or TXT files; content is split into chunks and stored in a FAISS vector index
- **Text paste & edit** — paste raw text, edit it later in-place, and re-index automatically
- **Retrieval-Augmented Generation (RAG)** — answers are grounded in your documents; the bot responds naturally when something is outside its knowledge
- **Streaming responses** — answers appear token by token as the LLM generates them
- **Configurable LLM** — OpenAI or Ollama; Ollama models are auto-detected from your local installation
- **Configurable embeddings** — OpenAI or a free local model (`all-MiniLM-L6-v2`) that runs on CPU
- **Per-session persona** — custom bot name and system prompt per session
- **Relevance threshold** — controls how strictly retrieved chunks are filtered
- **Export** — download a ZIP with a ready-to-deploy FastAPI server and a JavaScript chat widget
- **Chat history export** — download any conversation as JSON

---

## Project Structure

```
chatbot/
├── app.py                   Main Streamlit application
├── config.py                Central configuration (paths, defaults, env vars)
├── requirements.txt         Python dependencies
├── .env                     Your environment variables (edit this)
├── .streamlit/
│   └── config.toml          Streamlit server settings
└── data/
    ├── chat_history.db      SQLite database (auto-created on first run)
    ├── uploads/             Uploaded and pasted files
    ├── vector_stores/       FAISS indices, one folder per session
    └── core/
        ├── database.py      Sessions, messages, documents (SQLite)
        ├── document_processor.py  File loading and chunking
        ├── embeddings.py    Embedding model factory
        ├── export.py        Export ZIP builder
        ├── llm.py           LLM factory (OpenAI / Ollama)
        ├── rag_chain.py     RAG pipeline — retrieval, prompting, streaming
        └── vector_store.py  FAISS operations
```

---

## Configuration Reference

All settings live in the `.env` file:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `OPENAI_API_KEY` | — | Required when using OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any OpenAI chat model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `llama3.2` | Default Ollama model for new sessions |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `local` |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |

### Provider comparison

| | Ollama + local | OpenAI |
|---|---|---|
| Cost | Free | Pay per token |
| Privacy | 100% local | Data sent to OpenAI |
| Quality | Good | Excellent |
| Setup | Install Ollama + pull model | API key only |
| Speed | Depends on your hardware | Fast |

---

## Usage Guide

### Sessions

The sidebar starts with no session selected. Type a name and click **➕** to create one. Each session is fully isolated — its own documents, settings, and chat history. Switch between sessions using the dropdown at the top of the sidebar.

### Bot Settings

After creating a session, configure it in **⚙️ Bot Settings**:

- **Bot name** — how the bot introduces itself
- **System prompt** — defines the bot's role, tone, and domain focus
- **Model settings** expander:
  - **LLM provider** — switch between OpenAI and Ollama; the model dropdown updates automatically
  - **Embeddings** — `local` is free and runs on CPU; `openai` requires an API key
  - **Relevance threshold** — default `0.20`; lower it if the bot often says it doesn't know things

Always click **💾 Save settings** after making changes.

> ⚠️ If you change the **embedding provider** after indexing documents, you must delete the session and re-index. The old FAISS index is incompatible with the new embedding model.

### Adding Knowledge

Under **📄 Knowledge Base**, pick a mode:

**📁 Upload file** — supports PDF, DOCX, TXT. Select files and click **📥 Index files**. Files are split into 1 000-character overlapping chunks and embedded into FAISS.

**✏️ Paste text** — give the text a label (e.g. `Company FAQ`), paste the content, and click **📥 Index text**. The text is saved to disk as `label.txt`. Click the **✏️** button next to any `.txt` entry to edit it; saving re-indexes the whole session automatically.

### Chatting

Once documents are indexed, the chat input appears in the main area.

The bot:
- Answers greetings and questions about itself naturally
- Answers knowledge questions **only** from your indexed documents
- Responds specifically when something is outside its knowledge — e.g. *"I don't have information about pricing"*
- Clarifies it is an AI when asked personal questions (phone number, email, etc.) and offers to look up relevant info from the documents instead
- Streams the answer token by token
- Shows **📚 Sources** (filename, page, relevance %, snippet) below each answer

### Export

Under **📦 Export Bot**, click **⬇️ Download export package (ZIP)** to download a deployable package. See [Exporting](#exporting) below.

### Danger Zone

The **🗑️ Danger zone** expander at the bottom of the sidebar contains:
- **Clear chat history** — wipes all messages, keeps the knowledge base intact
- **Delete this session** — permanently removes the session, FAISS index, and all messages
- **Export chat (JSON)** — downloads the full conversation history

---

## Exporting

Export packages the trained bot as a standalone server you can deploy anywhere and embed on any webpage.

### Contents of the ZIP

| File | Description |
|---|---|
| `vector_store/` | FAISS index (your trained knowledge base) |
| `config.json` | Bot name, system prompt, model and embedding settings |
| `server.py` | Standalone FastAPI server with streaming `/chat` endpoint |
| `widget.html` | Chat widget with embeddable copy-paste snippet |
| `requirements.txt` | Python dependencies for the server |
| `.env.example` | Environment variable template |
| `README.md` | Deployment instructions |

### Running the Exported Server

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env — add OPENAI_API_KEY if you used OpenAI

# Start the server
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Testing the Widget

Open **http://localhost:8000** in your browser — the server serves the chat widget directly at the root URL.

> **Do not open `widget.html` as a local file.** Browsers block `fetch()` requests from `file://` pages to a local server, so the chat will not work. Always use `http://localhost:8000`.

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the chat widget UI |
| `POST` | `/chat` | Send a message, receive a streamed response |
| `GET` | `/health` | Returns `{"status": "ok", "bot": "..."}` |

**`POST /chat` body:**
```json
{
  "message": "What is the return policy?",
  "history": [
    {"role": "user",      "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help?"}
  ]
}
```
Response: plain-text stream (tokens arrive as they are generated).

### Embedding the Widget on Your Webpage

Open `widget.html` and copy everything between `<!-- WIDGET START -->` and `<!-- WIDGET END -->`. Paste it into your page's `<body>`. Then update the `API` constant to your **public** server URL:

```js
var API = "https://your-server.com/chat";
```

The widget renders as a floating chat bubble in the bottom-right corner.

### Hosting Options

| Platform | Notes |
|---|---|
| [Render](https://render.com) | Free tier, deploys from GitHub |
| [Railway](https://railway.app) | Free starter plan, zero config |
| [Fly.io](https://fly.io) | Generous free tier, global edge |
| Any VPS | Run `uvicorn` as a systemd service |

A Dockerfile example is included inside the exported `README.md`.

> **LLM notes after deploy:**
> - **OpenAI** — set `OPENAI_API_KEY` in `.env`. Works on any server, no GPU needed.
> - **Ollama** — requires Ollama installed on the deployment server. Pull the model first: `ollama pull llama3.2`

---

## Troubleshooting

**Ollama is not running**
```
Failed to establish a new connection: [WinError 10061] No connection could be made
```
Start Ollama in a separate terminal and keep it open:
```bash
ollama serve
```

**Ollama 404 — model not found**
```
Ollama call failed with status code 404
```
Pull the model first, then select it in Model settings and save:
```bash
ollama pull llama3.2
```

**`ModuleNotFoundError: No module named 'core'`**
You must run the app from the project root directory, not a subdirectory:
```bash
cd chatbot
streamlit run app.py
```

**Bot always says it doesn't have information**
- Lower the Relevance threshold slider in Model settings (try `0.10`)
- Confirm the documents were indexed in the current session (check the Knowledge Base list)
- If you changed the embedding provider after indexing, delete the session and re-index from scratch

**Slow first response with local embeddings**
`all-MiniLM-L6-v2` (~90 MB) downloads on first use. Subsequent runs load from cache and are fast.

**`Failed to fetch` in the exported widget**
Do not open `widget.html` as a file. Open **http://localhost:8000** in your browser instead.

**`Failed to fetch dynamically imported module` in the Streamlit app**
Hard-refresh the browser: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (macOS).

**OpenAI API errors**
Check that `OPENAI_API_KEY` in `.env` is correct and your account has credits at [platform.openai.com/usage](https://platform.openai.com/usage).

---

## Tech Stack

| Package | Role |
|---|---|
| `streamlit` | Web UI |
| `langchain` / `langchain-community` | RAG orchestration, document loaders |
| `langchain-openai` | OpenAI LLM and embeddings |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` + `torch` | Free local embedding model |
| `pypdf` | PDF parsing |
| `docx2txt` | DOCX parsing |
| `openai` | OpenAI API client |
| `python-dotenv` | `.env` file loading |
| `fastapi` + `uvicorn` | Exported chatbot server |
