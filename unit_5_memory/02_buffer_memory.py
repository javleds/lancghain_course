#!/usr/bin/env python
# coding: utf-8

import pickle
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI()

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# Ejemplo con RunnableWithMessageHistory:
# https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html
# Lanzamos el primer prompt (human message)
conversation.predict(
    input="Hola, necesito saber cómo usar mis datos históricos para crear un bot de preguntas y respuestas")
# Lanzamos el segundo prompt (human message)
conversation.predict(input="Necesito más detalle de cómo implementarlo")

print(memory.buffer)

# Cargamos la variable de memoria
memory.load_memory_variables({})

# Crea un objeto binario con todo el objeto de la memoria
pickled_str = pickle.dumps(conversation.memory)

# wb para indicar que escriba un objeto binario, en este caso en la misma ruta que el script
with open('memory.pkl', 'wb') as f:
    f.write(pickled_str)

# memoria_cargada = open('memory.pkl', 'rb').read()
#
#
# conversacion_recargada = ConversationChain(
#     llm=llm,
#     memory=pickle.loads(memoria_cargada),
#     verbose=True
# )
# print('\nReloaded conversation')
# print(conversacion_recargada.memory.buffer)
