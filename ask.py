from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import sys

ollama = Ollama(model="mistral:7b")

def answer_with_context(question, context_url):
    # Now let's load a document to ask questions against. 
    loader = WebBaseLoader(context_url)
    data = loader.load()

    # This file is pretty big. Which means the full document won't fit into the context for the model. So we need to split it up into smaller pieces.
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    # It's split up, but we have to find the relevant splits and then submit those to the model. 
    # We can do this by creating embeddings and storing them in a vector database. 
    # We can use Ollama directly to instantiate an embedding model. 
    oembed = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

    # And here is the relevant part of the document to the question.
    docs = vectorstore.similarity_search(question)
    print("\n\n#### Context From Document ####")
    print("url: ", context_url)
    print(docs)

    # The next thing is to send the question and the relevant parts of the docs to the model to see if we can get a good answer. 
    qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    output = qachain.invoke({"query": question})
    return output['result']

def answer(question):
    output = ollama.invoke(question)
    return output

def main():
    if len(sys.argv) < 2:
        print("Please provide a topic as an argument.")
        print('Usage 1: python ask.py "Who is current President in USA?"')
        print('Usage 2: python ask.py "Who is current President in USA?" "https://en.wikipedia.org/wiki/President_of_the_United_States"')

    # Get the first command-line argument after the script name
    question = sys.argv[1]

    print("\n\n#### Question ####")
    print(question)

    # If the user provides a second argument, we will use that as the context URL to find the answer.
    context_url = ''
    if len(sys.argv) == 3:
        context_url = sys.argv[2]

    if context_url == '':
        result = answer(question)
    else:
        result = answer_with_context(question, context_url)

    print("\n\n#### Result ####")
    print(result)

if __name__ == '__main__':
    main()