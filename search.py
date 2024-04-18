from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever

from argparse import ArgumentParser

# set up LlamaIndex retriever from saved index
def init_retriever(save_dir: str) -> VectorIndexRetriever:
    storage_context = StorageContext.from_defaults(persist_dir=save_dir)
    index = load_index_from_storage(storage_context)

    return VectorIndexRetriever(index=index, similarity_top_k=3)

def main():
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="index")
    
    args = parser.parse_args()
    
    embeddings = LangchainEmbedding(FastEmbedEmbeddings())
    Settings.embed_model = embeddings
    
    save_dir = args.save_dir
    retriever = init_retriever(save_dir)

    print("Now searching over list of Academy Award-winning films. Type 'exit' to quit.\n")

    while True:
        query = input("Enter query: ")
        if query == "exit":
            break

        results = retriever.retrieve(query)
        
        for result in results:
            print(f"{result.metadata["title"]}: {(result.score*100):.2f}% similar")
            print(result.text)
            print()
            
if __name__ == "__main__":
    main()