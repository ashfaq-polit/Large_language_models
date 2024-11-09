#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Ashfaq)s
"""

### Download model from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main
### https://medium.com/@fradin.antoine17/3-ways-to-set-up-llama-2-locally-on-cpu-part-1-5168d50795ac

# pip install llama-cpp-python

from llama_cpp import Llama


# Put the location of to the GGUF model that you've download from HuggingFace here
model_path = "/Users/adury/Desktop/llama-2-7b-chat.Q2_K.gguf"
llm = Llama(model_path=model_path)

# Prompt creation
system_message = "You are a helpful assistant"
user_message = "Q: Name the planets in the solar system? A: "

prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>
{user_message} [/INST]"""

# Run the model
output = llm(
  prompt, # Prompt
  max_tokens=128, # Generate up to 32 tokens
  stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
  echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion

# Output
print(output)

print(output["choices"][0]["text"])









