from chatbot import workflow
from langchain_core.messages import HumanMessage,AIMessage

thread_id="1"
while True:
    userMessage=input("Type Here: ")
    print("User: ",userMessage)


    if userMessage.strip().lower() in ['exit','quit','bye']:
        break

    config={"configurable":{"thread_id":thread_id}}
    response=workflow.invoke({"messages":[HumanMessage(content=userMessage)]},config=config)
    print("AI: ",response["messages"][-1].content)


