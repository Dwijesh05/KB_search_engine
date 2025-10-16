# KB Search Engine (Local RAG App)

A **local knowledge base search engine** that lets you upload multiple **PDF** or **TXT** files, summarizes them, and allows you to **query your documents** using **Retrieval-Augmented Generation (RAG)** — all **without any API key**.

Built with **Streamlit**, **SentenceTransformers**, **FAISS**, and **Transformers** — this app runs **completely offline** after initial model downloads.

---

## 🏗 Repository Structure

```
KB_search_engine/
├── knowledge_base_app.py
├── README.md
└── (other modules, config, etc.)
```

* **knowledge_base_app.py** — main Streamlit application
* (You may also have files for utility modules, embeddings, summarization, etc.)

---

## 🚀 Features

* Upload multiple **PDF** or **TXT** documents
* Automatic **summarization** of long documents
* Convert documents to **embeddings** using open-source models
* Local **vector search** via FAISS
* **Offline and private** — no API key or external service needed
* Simple and clean **Streamlit UI**

---

## 🛠️ Installation & Setup

### Step 1: Clone or Download

```bash
git clone https://github.com/Dwijesh05/KB_search_engine.git
cd KB_search_engine
```

If you downloaded a ZIP, extract it and then `cd` into the folder.

---

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv kb_env
```

Activate it:

* **On Windows:**

  ```bash
  kb_env\Scripts\activate.bat
  ```

* **On macOS / Linux:**

  ```bash
  source kb_env/bin/activate
  ```

---

### Step 3: Install Dependencies

```bash
pip install streamlit torch transformers sentence-transformers faiss-cpu pymupdf
```

---

## ▶️ Step 4: Run the App

```bash
streamlit run knowledge_base_app.py
```

This will start the app locally.

---

## ⚙️ First-Time Initialization

When you run it for the first time:

* The models will **download automatically**
* You’ll see progress bars for downloading:

  * **`all-MiniLM-L6-v2`** — for embeddings
  * **`sshleifer/distilbart-cnn-12-6`** — for summarization

Example output:

```
Downloading: 100%|██████████| 90.9M/90.9M [00:05<00:00]
Downloading: 100%|██████████| 1.42G/1.42G [00:30<00:00]
```

---

## 🌐 Accessing the App

After launching:

```
Local URL: http://localhost:8501
Network URL: http://<your-machine-IP>:8501
```

Open `http://localhost:8501` in your browser to use the app.

---

## 🔄 Typical Workflow

1. Start the app with `streamlit run knowledge_base_app.py`
2. Upload one or more **PDF** or **TXT** docs
3. Wait for summarization & embeddings generation
4. Enter queries in the search/query box
5. Receive answers based on your document content

---

## 📋 Dependencies & Versions

* Python 3.8+
* streamlit
* torch
* transformers
* sentence-transformers
* faiss-cpu
* pymupdf

You can also pin versions in a `requirements.txt` if needed.

---

## 🧠 Models Used

| Task            | Model                           | Source               |
| --------------- | ------------------------------- | -------------------- |
| Text Embeddings | `all-MiniLM-L6-v2`              | SentenceTransformers |
| Summarization   | `sshleifer/distilbart-cnn-12-6` | Hugging Face         |

---


