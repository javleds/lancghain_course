#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías iniciales e instancia de modelo de chat

# In[ ]:


from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
f = open('../OpenAI_key.txt')
api_key = f.read()
llm = ChatOpenAI(openai_api_key=api_key)


# #  Crear objeto ConversationSummaryMemory

# In[ ]:


memory = ConversationSummaryBufferMemory(llm=llm)


# In[ ]:


# Creamos un prompt cuya respuesta hará que se sobrepase el límite de tokens y por tanto sea recomendable resumir la memoria
plan_viaje = '''Este fin de semana me voy de vacaciones a la playa, estaba pensando algo que fuera bastante relajado, pero necesito 
un plan detallado por días con qué hacer en familia, extiéndete todo lo que puedas'''


# ## Creamos una nueva conversación con un buffer de memoria resumida

# In[ ]:


memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
conversation = ConversationChain(llm=llm,memory = memory,verbose=True)
#Ejemplo con RunnableWithMessageHistory: https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html


# In[ ]:


conversation.predict(input=plan_viaje)


# In[ ]:


memory.load_memory_variables({}) #Se ha realizado un resumen de la memoria en base al límite de tokens


# In[ ]:


print(memory.buffer)

