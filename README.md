# Customer Support Bot

A small-business customer support chatbot that answers questions **only from your own documents** (FAQs, policies, guides). The business owner uploads PDF, TXT or DOCX files on an admin page; customers ask questions on a chat page and get short, friendly answers with the source document shown.

It uses **Retrieval-Augmented Generation (RAG)**: the bot first finds the most relevant passages in your documents, then asks Google Gemini to answer using only those passages. If nothing relevant is found, it says it doesn't know instead of guessing.

## Features

- Chat page that works on phones and desktops, with a typing indicator, clickable example questions and a "Sources" tag under every answer
- Admin page with drag-and-drop upload (multiple PDF, TXT, DOCX files), upload progress, per-file results, and a document list with delete buttons
- Answers use only your documents; questions outside them get a polite "please contact the business" reply, without calling the AI model at all
- Re-uploading a file replaces its old version (no duplicates)
- Optional admin password
- Automatic retries with backoff for Gemini rate limits and temporary errors
- Friendly error messages everywhere; stack traces are never shown to users
- No build step: plain HTML, CSS and JavaScript served by the same FastAPI server

## Architecture

```
   Customer browser                          Admin browser
   index.html + chat.js                      admin.html + admin.js
          |                                         |
          | POST /api/chat                          | POST /api/ingest
          |                                         | GET /api/documents
          |                                         | DELETE /api/documents/{name}
          v                                         v
   +--------------------------------------------------------------+
   |                 FastAPI server  (backend/main.py)            |
   |                                                              |
   |   llm.py                           rag.py                    |
   |   - build the prompt               - extract text            |
   |   - call Gemini                    - chunk                   |
   |   - return answer + sources        - embed                   |
   |                                    - store / retrieve        |
   +--------------------+-------------------------+---------------+
                        |                         |
                        v                         v
                 Google Gemini API        ChromaDB (./chroma_db)
              (embeddings + answers)       local vector database
```

## How the RAG flow works

**Ingest (admin uploads a file)**

1. **Extract**: read the text from the PDF (`pypdf`), DOCX (`python-docx`) or TXT file.
2. **Chunk**: split the text into ~500-character pieces with a 50-character overlap, breaking at sentence or paragraph boundaries so each chunk reads naturally.
3. **Embed**: send the chunks to Gemini (`gemini-embedding-001`, task type `RETRIEVAL_DOCUMENT`) in batches of up to 50. Each chunk becomes a list of numbers (a vector) that captures its meaning.
4. **Store**: save the chunk text, its vector and its source filename in ChromaDB. IDs look like `<filename>::<chunk_index>`, so uploading the same file again replaces the old chunks.

**Answer (customer asks a question)**

5. **Retrieve**: embed the question (task type `RETRIEVAL_QUERY`) and ask ChromaDB for the 4 closest chunks. Cosine distance is converted to similarity (`1 - distance`) and chunks scoring below `0.35` are dropped.
6. **Generate**: if no chunk passed the threshold, reply "I don't have information on that" and stop (no model call). Otherwise send Gemini the system instructions, the retrieved chunks, the last 4 chat messages and the new question, at temperature 0.2. The answer is returned with a de-duplicated list of source filenames.

## Tech stack and why

| Part | Choice | Why |
|------|--------|-----|
| Backend | Python + FastAPI + uvicorn | Simple, fast, easy to read, validates input, serves the frontend too |
| Vector database | ChromaDB (persistent, local) | No separate server to run; stores data in a folder; built-in cosine search |
| LLM and embeddings | Google Gemini via the official `google-genai` SDK | One API key for both; a free tier is available; the SDK is the current supported one |
| Frontend | Plain HTML, CSS, vanilla JS | No build step, nothing to install, loads instantly, easy to customise |
| File parsing | `pypdf`, `python-docx` | Pure-Python, no system packages needed |
| Config | `python-dotenv` | Reads a local `.env` file; Replit Secrets work as normal environment variables |

## Project structure

```
customer-support-bot/
  backend/
    main.py            API routes, validation, static files, server start
    rag.py             extraction, chunking, embedding, ChromaDB, retrieval
    llm.py             prompt building and Gemini answering
    config.py          all settings (models, chunk sizes, limits)
    requirements.txt   Python dependencies
    .env.example       template for your secrets
  frontend/
    index.html, chat.js     customer chat page
    admin.html, admin.js    document upload page
    style.css               shared styles
  sample_docs/         two generic example documents for a small online shop
  .replit              makes the Run button work on Replit
  .gitignore
  README.md
```

## Setup

1. Create a new Repl and import this project (upload the folder, or import it from GitHub).
2. Open **Secrets** (the padlock icon) and add:
   - `GEMINI_API_KEY`: your key from <https://aistudio.google.com/apikey>
   - `ADMIN_PASSWORD` (optional): protects the admin page
3. Click **Run**. The first run installs the dependencies, then starts the server on port 8000.
4. Open the web preview. The chat page is at `/` and the admin page is at `/admin`.
5. On the admin page, upload the two files from `sample_docs/` (or your own documents), then go back to the chat and try the example questions.

Note: the `chroma_db` folder lives on the Repl's disk. On a Reserved VM deployment it persists; on Autoscale deployments the disk is temporary, so documents would need to be uploaded again after a restart.

## Setup: local computer (Windows, Mac, Linux)

You need Python 3.10 or newer.

**1. Open a terminal in the project folder, then create a virtual environment**

```bash
# Mac / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

**2. Install the dependencies**

```bash
pip install -r backend/requirements.txt
```

**3. Add your secrets**

```bash
# Mac / Linux
cp backend/.env.example .env

# Windows
copy backend\.env.example .env
```

Open `.env` and set `GEMINI_API_KEY=` to your key. The other lines can stay empty.

**4. Start the app**

```bash
python backend/main.py
```

Open <http://localhost:8000> for the chat and <http://localhost:8000/admin> to upload documents.

## Settings

Edit `backend/config.py`, or set environment variables for the model names:

| Setting | Default | Meaning |
|---------|---------|---------|
| `GEN_MODEL` (env) | `gemini-2.5-flash` | Model that writes answers |
| `EMBED_MODEL` (env) | `gemini-embedding-001` | Model that creates embeddings |
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Characters shared between neighbouring chunks |
| `TOP_K` | 4 | Chunks retrieved per question |
| `MIN_SIMILARITY` | 0.35 | Chunks scoring lower than this are ignored |
| `MAX_HISTORY_TURNS` | 4 | Recent chat messages sent to the model |

The three clickable example questions are plain buttons in `frontend/index.html`; change them to match your own documents.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Returns `status`, whether `GEMINI_API_KEY` is set, and whether the admin password is enabled (never the key itself) |
| `POST` | `/api/chat` | Body `{"message": "...", "history": [{"role": "user" or "assistant", "content": "..."}]}`; returns `{"answer": "...", "sources": ["file.txt"]}` |
| `POST` | `/api/ingest` | Multipart upload with one or more `files`; returns per-file results (`chunks` added or an `error`). Admin |
| `GET` | `/api/documents` | Lists indexed documents with chunk counts. Admin |
| `DELETE` | `/api/documents/{filename}` | Removes a document. Admin |

"Admin" endpoints require the header `X-Admin-Password` only if the `ADMIN_PASSWORD` secret is set. Errors are always JSON in the form `{"error": "message"}`. Limits: messages up to 1000 characters, files up to 10 MB.

## Troubleshooting

- **"The server is missing its GEMINI_API_KEY"**: add the key to Replit Secrets or your `.env` file and restart. Check `/api/health`: `gemini_key_set` should be `true`.
- **"The Gemini model was not found"**: Google renames and retires models. Set `GEN_MODEL` or `EMBED_MODEL` to a current model name from the Gemini API docs, then restart.
- **"The AI service is busy or the free quota has been reached"**: the free tier has tight rate limits. Wait a minute and retry. Uploading many large files at once is the most common cause.
- **The bot says it doesn't know something that is in your documents**: the similarity threshold may be too strict. The server log prints the score of every retrieved chunk; if good matches score just under `0.35`, lower `MIN_SIMILARITY` a little.
- **The bot answers questions that are not in your documents**: raise `MIN_SIMILARITY`, or check the log scores of an unrelated question and set the threshold just above them.
- **"No readable text was found"** when uploading a PDF: it is probably a scan (images only). Use a text-based PDF, or run OCR on it first.
- **Port already in use (local)**: set a different port, for example `PORT=8001 python backend/main.py` (on Windows PowerShell: `$env:PORT=8001; python backend/main.py`).
- **Start fresh**: stop the app and delete the `chroma_db` folder to remove all indexed documents.

## Future improvements

- Stream answers word by word as they are generated
- Use the previous question when retrieving, so follow-ups like "and for international orders?" work better
- OCR support for scanned PDFs, plus HTML and Markdown files
- Re-ranking of retrieved chunks for better accuracy
- Per-user admin accounts instead of one shared password
- Rate limiting and basic analytics on the chat endpoint (common questions, unanswered questions)
- A "was this helpful?" button, and hand-off to a human or an email form for unanswered questions
- Multi-business support, with a separate document collection for each business

<img width="932" height="862" alt="Screenshot 2026-11-01 144729" src="https://github.com/user-attachments/assets/70be5184-33f0-4948-95df-73416cbe1e2b" />
<img width="963" height="862" alt="image" src="https://github.com/user-attachments/assets/ce4f2a93-0e4e-4ad4-9abd-e533280ff461" />


