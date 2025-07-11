# -*- coding: utf-8 -*-
"""RAG_with_Conversation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/sunnysavita10/Generative-AI-Indepth-Basic-to-Advance/blob/main/RAG_with_Conversation.ipynb
"""

# --- Install required libraries ---
!pip install -q langchain langchain-community langchainhub langchain-chroma sentence-transformers accelerate bitsandbytes

!pip uninstall -y transformers accelerate bitsandbytes
!pip install --upgrade --quiet transformers accelerate bitsandbytes

# --- 1. Setup Hugging Face token and LLaMA-3 model ---
from google.colab import userdata
hf_token = userdata.get("llama3")

import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    torch_dtype=torch.float16,
    device_map="auto"
)

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=llama_pipeline)

from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4

loader = WebBaseLoader(
    web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

# Prompts
system_prompt = (
    "You are a helpful assistant. Use the retrieved context to answer the question. "
    "If you don't know the answer, say so. Be concise (max 3 sentences).\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
retriever_prompt = (
    "Given the chat history and latest user question, rephrase it into a standalone question if needed."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", retriever_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# RAG chain
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

session_id = "demo-session"

# First question
response1 = conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={"configurable": {"session_id": session_id}}
)
print("Q1:", response1["answer"])

# Second question
response2 = conversational_rag_chain.invoke(
    {"input": "What are common ways of doing it?"},
    config={"configurable": {"session_id": session_id}}
)
print("Q2:", response2["answer"])