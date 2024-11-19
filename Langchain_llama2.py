#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Ashfaq)s
"""

### Download model from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main

# pip install llama-cpp-python

from llama_cpp import Llama

# Put the location of to the GGUF model that you've download from HuggingFace here
model_path = "/Users/adury/Desktop/llama-2-7b-chat.Q2_K.gguf"

## https://lunary.ai/blog/llama-cpp-python#creating-a-simple-llm-chain
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

# Set up callback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Create the LLM object
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=200,
    top_p=1.0,
    callback_manager=callback_manager,
    verbose=True
)

# Example usage
question = "What is bindings in programming languages?"
response = llm(question)
print(response)


### https://www.mlexpert.io/blog/langchain-quickstart-with-llama-2
### Prompts and Prompt Templates
from langchain import PromptTemplate
 
template = """
<s>[INST] <<SYS>>
Act as a Machine Learning engineer who is teaching high school students.
<</SYS>>
 
{text} [/INST]
"""
 
prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)

text = "Explain what are Deep Neural Networks in 2-3 sentences"
print(prompt.format(text=text))

result = llm(prompt.format(text=text))
print(result)


### Create a Chain
from langchain.chains import LLMChain
 
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(text)
print(result)

template = "<s>[INST] Use the summary {summary} and give 3 examples of practical applications with 1 sentence explaining each [/INST]"
 
examples_prompt = PromptTemplate(
    input_variables=["summary"],
    template=template,
)

examples_chain = LLMChain(llm=llm, prompt=examples_prompt)


from langchain.chains import SimpleSequentialChain
 
multi_chain = SimpleSequentialChain(chains=[chain, examples_chain], verbose=True)
result = multi_chain.run(text)
print(result.strip())


### Chatbot
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage
 
template = "Act as an experienced high school teacher that teaches {subject}. Always give examples and analogies"
human_template = "{text}"
 
chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(template),
        HumanMessage(content="Hello teacher!"),
        AIMessage(content="Welcome everyone!"),
        HumanMessagePromptTemplate.from_template(human_template),
    ]
)
 
messages = chat_prompt.format_messages(
    subject="Artificial Intelligence", text="What is the most powerful AI model?"
)
messages

result = llm.predict_messages(messages)
print(result.content)





