import requests
from bs4 import BeautifulSoup

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from itertools import batched
from urllib.parse import unquote
from argparse import ArgumentParser

WIKI_BASE_URL = "https://en.wikipedia.org/w/api.php"

# get all pages linked in wikipedia page table
def get_wiki_pages(title: str) -> list:
    params = {
        'action': 'parse',
        'format': 'json',
        'page': title,
        'formatversion': 2
    }

    response = requests.get(WIKI_BASE_URL, params=params)
    data = response.json()

    html = data['parse']['text']
    parser = BeautifulSoup(html, 'html.parser')

    results = parser.select('table.wikitable > tbody > tr > td > i > a[href]')    
    pages = [unquote(result.get('href').removeprefix("/wiki/")) for result in results]

    print(f"Found {len(pages)} pages")

    return pages

# get only page intros (faster, can be done in batches)
def get_pages_intros(titles: list) -> list:
    docs = []
    batch_size = 20
    
    for titles in batched(titles, batch_size):
        params = {
            'action': 'query',
            'format': 'json',
            'titles': '|'.join(titles),
            'prop': 'extracts|description',
            'formatversion': 2,
            'explaintext': True,
            'exintro': True
        }
        
        response = requests.get(WIKI_BASE_URL, params=params)
        
        for page in response.json()['query']['pages']:
            try:
                docs.append(Document(
                    extra_info={
                        'title': page['title'],
                        'description': page['description']
                    },
                    text=page['extract']
                ))
            except KeyError:
                print(f"No extract found for {page['title']}")
        
        print(f"Processed {batch_size} pages")
        
    return docs

# get full page content for all pages (slower)
def get_pages_full(titles: list) -> list:
    docs = []
    
    for title in titles:
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts|description',
            'formatversion': 2,
            'explaintext': True
        }
        
        response = requests.get(WIKI_BASE_URL, params=params)
        page = response.json()['query']['pages'][0]
        
        try:
            docs.append(Document(
                extra_info={
                    'title': page['title'],
                    'description': page['description']
                },
                text=page['extract']
            ))
            print(f"Processed {page['title']}")
        except KeyError:
            print(f"No extract found for {page['title']}")
            
    return docs

# create LlamaIndex index from list of documents
def create_index_llamaindex(docs: list, save_dir: str = "index"):
    embed_model = LangchainEmbedding(FastEmbedEmbeddings())

    Settings.embed_model = embed_model
    Settings.transformations = [SentenceSplitter(chunk_size=512, chunk_overlap=20)]

    index = VectorStoreIndex.from_documents(docs, show_progress=True)

    index.storage_context.persist(persist_dir=save_dir)

    return index

def main():
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="index")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--title", type=str, default="List_of_Academy_Award–winning_films")
    
    args = parser.parse_args()
    
    pages = get_wiki_pages(args.title)
    
    if args.full:
        docs = get_pages_full(pages)
    else:
        docs = get_pages_intros(pages)
    
    create_index_llamaindex(docs, save_dir=args.save_dir)
    
if __name__ == "__main__":
    main()