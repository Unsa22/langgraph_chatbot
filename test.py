from rag import get_retriever, format_docs, load_chunks, load_vs
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableSequence
from prompts import template
from model import model
from langchain_core.output_parsers import StrOutputParser
from state import stateChatbot


config={"run_name":"pcos_chatbot"}

retriever= get_retriever(load_vs("./db"), k=3, chunks= load_chunks())


parallel_chain= RunnableParallel({
    "context":retriever | RunnableLambda(format_docs),
    "question":RunnablePassthrough()
})

chain= RunnableSequence(steps=[ parallel_chain , template , model , StrOutputParser()])
print("Ask Question: ")
q= input()
ans= chain.invoke(q, config=config)
print("\n\n", ans)



