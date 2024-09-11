#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI()

# Creamos un prompt cuya respuesta hará que se sobrepase el límite de tokens y por tanto sea recomendable resumir la memoria
plan_viaje = '''Este fin de semana me voy de vacaciones a la playa, estaba pensando algo que fuera bastante relajado, pero necesito 
un plan detallado por días con qué hacer en familia, extiéndete todo lo que puedas'''

# Ejemplo con RunnableWithMessageHistory:
# https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
conversation.predict(input=plan_viaje)
memory.load_memory_variables({})  # Se ha realizado un resumen de la memoria en base al límite de tokens

print('\nMemory buffer:')
print(memory.buffer)
