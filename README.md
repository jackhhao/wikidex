# wikidex

Fast + local semantic search over select body of Wikipedia pages.

## Installation

Using pip:

```bash
pip install -r requirements.txt
```

## Usage
To construct the search index:
```bash
python construct_index.py --save_dir=<directory name> --title=<title of main Wikipedia list>
```

To query over the index:
```bash
python search.py --save_dir=<directory name>
```

## Architecture
I chose to use prebuilt LLM orchestration tools such as LangChain and LlamaIndex for their efficiency and simplistic usage. Saves having to manually implement text chunking, embedding, and vector lookup via an algorithm like FAISS.

I also chose to use the open-source FastEmbed embeddings model (defaults to `BAAI/bge-base-en` on HuggingFace). This prioritizes latency and end-user experience while maintaining accuracy of the lookup.

A command line interface was chosen for simplicity and ease of implementation.