# CS311 Handbook RAG

A retrieval-augmented generation (RAG) chatbot that answers questions about the **B.S. Computer Science major handbook**. Everything runs locally: the documents, the vector database and the language models stay on your computer.

Final project for CS311 Advanced Deep Learning, Fall 2026.

## Features

- **PDF support**: reads a PDF page by page, repairs extraction errors (broken ligatures, `Descrip8on`-style glitches) and remembers the page of every chunk.
- **Structure-aware chunking**: one chunk per course description, and separate chunks for the Major and Minor requirement lists.
- **Grounded answers**: the model may only use the retrieved text, and replies *"The handbook does not say."* when the answer is missing.
- **Fallback for unrelated questions**: if the retrieved text is not relevant enough, the model is not called at all.
- **Sources**: every answer shows the pages it came from.
- **Three ways to use it**: command line, web interface (Streamlit), and Docker Compose.

## How it works

```
PDF -> extract and clean -> chunk (with page) -> embed -> ChromaDB
Question -> retrieve top 5 -> score check -> prompt + LLM -> answer + pages
```

Embeddings use `nomic-embed-text`. The answer model is `qwen2.5` (3B by default; 1.5B and 0.5B were also compared). Models are served by [Ollama](https://ollama.com).

## Project structure

```
.
├── run_rag.py               # RAG pipeline and command-line interface
├── app.py                   # Streamlit web interface
├── questions.txt            # the 16 test questions (category | question)
├── major_handbook_pdfs/     # CS_MajorHandbook.pdf
├── evaluation/              # hand-graded results (graded_*.csv)
├── results/                 # raw outputs of each evaluation run
├── Dockerfile
├── docker-compose.yml
├── requirements-docker.txt  # pinned dependencies
├── DEPLOY.md                # short Docker guide
└── README.md
```

## Requirements

- Python 3.12 (the version used in the Docker image)
- [Ollama](https://ollama.com) with the models `nomic-embed-text` and `qwen2.5:3b`
- Python packages, pinned in `requirements-docker.txt`:

| Package | Version |
|---|---|
| langchain-core | 1.8.1 |
| langchain-text-splitters | 1.1.3 |
| langchain-chroma | 1.1.0 |
| langchain-ollama | 1.1.0 |
| chromadb | 1.5.9 |
| pypdf | 5.9.0 |
| streamlit | 1.67.0 |

## Installation (without Docker)

Clone the repository (use the URL from the green **Code** button on GitHub), then:

```bash
cd cs311-handbook-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-docker.txt

ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

Ollama must be running (`ollama serve`) while you use the chatbot. The first question builds the vector store from the PDF; later runs reuse it.

## Usage

Ask one question:

```bash
python run_rag.py --question "What is CS 203 about?" --model qwen2.5:3b
```

Chat with several questions in a row (type `exit` to quit):

```bash
python run_rag.py --chat --model qwen2.5:3b
```

Show the score and the retrieved chunks:

```bash
python run_rag.py --question "What is CS 203 about?" --debug
```

Rebuild the vector store (needed after changing the chunking code):

```bash
python run_rag.py --reindex --question "What is CS 203 about?"
```

Run the whole test set and save the answers to a CSV file:

```bash
python run_rag.py --questions questions.txt --out results/my_run.csv --model qwen2.5:3b
```

Start the web interface at http://localhost:8501:

```bash
streamlit run app.py
```

### Example

```
Q: How many advanced courses must a CS student take?
A: Four advanced courses.
Sources: page 7 (score=0.6695), page 11 (score=0.6645), page 7 (score=0.6476), page 11 (score=0.6457), page 7 (score=0.6447)
```

### Settings

Environment variables override the defaults:

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | address of the Ollama service |
| `LLM_MODEL` | `qwen2.5:1.5b` | answer model |
| `PDF_PATH` | `major_handbook_pdfs/CS_MajorHandbook.pdf` | document to index |
| `VECTOR_STORE` | `vector_store_cs` | folder of the vector database |

The score threshold (0.52), the number of retrieved chunks (5), the temperature (0.1) and the output limit (300 tokens) are constants at the top of `run_rag.py`.

## Docker deployment

Docker Compose starts an Ollama container and the chatbot. The PDF, the vector database and the results are mounted from your computer, and the downloaded models are kept in a Docker volume, so nothing is lost when containers are removed.

```bash
mkdir -p vector_store_docker results
docker compose build
docker compose up -d ollama
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen2.5:3b
```

Ask one question, chat, or run the whole test set:

```bash
docker compose run --rm rag --question "What is CS 203 about?"
docker compose run --rm rag --chat
docker compose run --rm rag --questions questions.txt --out results/docker_qwen2.5-3b.csv
```

Start the web interface and open http://localhost:8501:

```bash
docker compose up -d ui
```

Stop everything (models and index are kept; add `-v` to also delete the models):

```bash
docker compose down
```

Notes:

- Create `vector_store_docker` and `results` before the first run, otherwise Docker creates them as root-owned folders.
- The models are not inside the image. They are downloaded once, about 2.2 GB in total.
- To use another model, run for example `LLM_MODEL=qwen2.5:1.5b docker compose run --rm rag --chat` after pulling it.
- The Ollama port is not published to the host, so it does not clash with an Ollama already running on your computer.

## Evaluation

The chatbot was tested on 16 questions (requirements, courses, learning outcomes, information not in the handbook, and unrelated questions). Every answer was graded by hand against the handbook. The graded files are in `evaluation/`.

| Run | Correct | Incomplete | Ambiguous | Incorrect |
|---|---|---|---|---|
| First version, 0.5B | 2 | 3 | 0 | 11 |
| 1.5B | 8 | 1 | 1 | 6 |
| 3B, first chunking | 10 | 2 | 1 | 3 |
| 3B, final chunking | 10 | 2 | 0 | 4 |

What the results showed:

- Most failures were reading errors: the right page was retrieved, but the small model picked the wrong sentence.
- Relevance scores cannot tell an unanswerable question from an answerable one, so the prompt must handle refusals. The score check only catches completely unrelated questions.
- With 16 questions, a difference of one or two answers between runs is noise.

The full analysis is in the project report.

## Known limitations

- **Small test set**: 16 questions, graded by one person.
- **Page 10 is not indexed**: it is a flowchart image, and OCR is not implemented.
- **Tables are flattened** by the PDF text extraction, which weakens the link between a course and its prerequisites.
- **Hallucination is not fully solved**: small models sometimes answer questions that the handbook does not cover (for example, double major).
- **No conversation memory**: each question is answered on its own, so follow-up questions are not understood.
- **CPU only**: the first answer after a model is loaded can take several minutes.
- **Chunk counts depend on the `pypdf` version**: the pinned version gives the same chunking on every machine.
- **Windows with WSL**: Docker and Ollama models are stored inside the WSL disk. A model download failed when the system drive was almost full, so keep several GB free or move the WSL installation to another drive.
