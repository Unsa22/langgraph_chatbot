from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from model import model
from langchain.chains import RetrievalQA
import warnings
import os
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pickle
from langsmith import Client,traceable
import logging
warnings.filterwarnings("ignore")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.ERROR)
client = Client()
print("langsmith connected", client)

config = {"run_name": "pcos rag"}

@traceable(name="sc")
def save_chunks(chunks):
    with open("./chunks/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("file saved")


@traceable(name="lc")
def load_chunks():
    with open("./chunks/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print("chunk loaded")
    return chunks
@traceable(name="doe")
def delete_old_embeddings():
    if os.path.exists("./db"):
        for file in os.listdir("./db"):
            os.remove(os.path.join("./db", file))
        print("Old embeddings deleted.")
@traceable(name="loader")
def loader(path: str):
    dir = DirectoryLoader(
        path=path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    docs = []
    for i in dir.lazy_load():
        docs.append(i)
    return docs
@traceable(name="split")
def split(documents, cs, co):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=co
    )
    return chunks.split_documents(documents)
@traceable(name="vs")
def vectorestore(chunks, path):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name="pcos_fyp",
        persist_directory=path
    )
    vs.persist()
    return vs
@traceable(name="lvc")
def load_vs(path):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = Chroma(
        persist_directory=path,
        collection_name="pcos_fyp",
        embedding_function=embeddings
    )
    return vs
@traceable(name="fd")
def format_docs(docs):
    return "\n\n".join(i.page_content for i in docs)
@traceable(name="gr")
def get_retriever(vs, chunks, k):
    base_retriever = vs.as_retriever(search_kwargs={"k": k})
    base_retriever2 = BM25Retriever.from_documents(chunks)
    hybird_retriever = EnsembleRetriever(
        retrievers=[base_retriever, base_retriever2],
        weights=[0.5, 0.5]
    )
    multi_query = MultiQueryRetriever.from_llm(
        retriever=hybird_retriever,
        llm=model,

    )
    return multi_query
@traceable(name="rq")
def run_query(retriever, query):
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="stuff",
        
    )
    result = qa_chain.invoke({"query": query}, config=config)

    print("\nAI Answer:\n", result["result"])
   
@traceable(name="fs")
def from_start(path, query):
    docs = loader(path)
    chunks = split(docs, cs=1000, co=100)
    save_chunks(chunks=chunks)
    delete_old_embeddings()
    vs = vectorestore(chunks, "./db")
    retriever = get_retriever(vs, chunks, k=5)
    run_query(retriever, query)
@traceable(name="m")
def merge(path, query):
    if os.path.exists("./chunks"):
        chunks = load_chunks()
        if os.path.exists("./db"):
            vs = load_vs("./db")
            retriever = get_retriever(vs, chunks, k=5)
            run_query(retriever, query)
        else:
            from_start(path, query)
    else:
        print("path not available")

# merge(path="./knowledge_base", query="What is PCOS?")



