#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI()

# k indica el número de iteraciones (pareja de mensajes human-AI) que guardar en la memoria
memory = ConversationBufferWindowMemory(k=1)

# Ejemplo con RunnableWithMessageHistory:
# https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
conversation.predict(input="Hola, ¿cómo estás?")
conversation.predict(input="Necesito un consejo para tener un gran día")

print('\nMemory buffer:')
print(memory.buffer)
