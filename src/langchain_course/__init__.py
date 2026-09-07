import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")

    print("Loading the document...")
    loader = UnstructuredLoader(
        file_path="/Users/adeo/code/emarco-177/langchain-course/medium-blog-vector-db.txt",
        chunking_strategy="basic",
        max_characters=1000000,
    )
    document = loader.load()

    print("Splitting document...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    print("Ingesting embeddings...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ.get("INDEX_NAME")
    )


if __name__ == "__main__":
    main()
