# LLM Document Processing Pipeline

Fault-tolerant automation pipeline for ingesting unstructured documents through LLM APIs, extracting structured outputs with validated accuracy at scale.

## Tech Stack

- **Language**: Python 3.11+
- **LLM**: OpenAI API (GPT-4o)
- **PDF**: PyPDF2
- **Validation**: Pydantic

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Run pipeline
python -m doc_pipeline run path/to/document.pdf
```

## Architecture

```
ingestion → inference → validation → storage
```

## Supported File Types

- `.txt` - Plain text
- `.md` - Markdown
- `.json` - JSON
- `.csv` - CSV
- `.pdf` - PDF (via PyPDF2)

## Project Structure

```
doc-pipeline/
├── src/doc_pipeline/
│   └── ingestion.py    # Document loading & chunking
├── tests/
└── README.md
```