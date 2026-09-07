import os

from dotenv import load_dotenv

load_dotenv()

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


print("Initializing Components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()
vector_store = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"),
    embedding=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    Provide a detailed answer:"""
)


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    relevant_docs = retriever.invoke(query)
    context = format_docs(relevant_docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)

    return response.content


def retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


def main() -> None:
    print("Retrieving...")

    query = "What is Pinecone in machine learning?"

    # Option 1: Without LCEL
    print("Retrieving without LCEL...")
    response_without_lcel = retrieval_chain_without_lcel(query)
    print(f"{query}:\n\n{response_without_lcel}")

    # Option 2: With LCEL
    print("\n\n\nRetrieving with LCEL...")
    response_with_lcel = retrieval_chain_with_lcel().invoke({"question": query})
    print("Retrieving with LCEL...")
    print(response_with_lcel)


if __name__ == "__main__":
    main()
