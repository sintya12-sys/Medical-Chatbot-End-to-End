
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
# You are an assistant for question-answering tasks. 
# Use the retrieved context below to answer the question. 

# Use at most three sentences.

# Context:
# {context}

# Question: {input}
# """)
