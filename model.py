from langchain_openai import ChatOpenAI

from dotenv import load_dotenv 
import os
load_dotenv()
model=ChatOpenAI(
    model="mistralai/mistral-small-3.2-24b-instruct:free",
    temperature=.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
)
