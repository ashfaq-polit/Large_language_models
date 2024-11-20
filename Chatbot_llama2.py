#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Ashfaq)s
"""

### Download model from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main
### https://medium.com/@weidagang/hello-llm-building-a-local-chatbot-with-langchain-and-llama2-3a4449fc4c03

# pip install llama-cpp-python
# pip install langchain

from llama_cpp import Llama

# Put the location of to the GGUF model that you've download from HuggingFace here
model_path = "/Users/adury/Desktop/llama-2-7b-chat.Q2_K.gguf"

## https://lunary.ai/blog/llama-cpp-python#creating-a-simple-llm-chain
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load the LlamaCpp language model, adjust GPU usage based on your hardware
llm = LlamaCpp(
    model_path="/Users/adury/Desktop/llama-2-7b-chat.Q2_K.gguf",
    # n_gpu_layers=40,
    n_batch=512,  # Batch size for model processing
    verbose=False,  # Enable detailed logging for debugging
)


# Define the prompt template with a placeholder for the question
template = """
Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create an LLMChain to manage interactions with the prompt and model
llm_chain = LLMChain(prompt=prompt, llm=llm)


print("Chatbot initialized, ready to chat...")
while True:
    question = input("> ")
    answer = llm_chain.run(question)
    print(answer, '\n')



