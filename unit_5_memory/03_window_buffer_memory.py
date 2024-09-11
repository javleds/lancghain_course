#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías iniciales e instancia de modelo de chat

# In[ ]:


from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
f = open('../OpenAI_key.txt')
api_key = f.read()
llm = ChatOpenAI(openai_api_key=api_key)


# #  Crear objeto ConversationBufferWindowMemory

# In[ ]:


memory = ConversationBufferWindowMemory(k=1) #k indica el número de iteraciones (pareja de mensajes human-AI) que guardar


# ## Conectar una conversación a la memoria

# In[ ]:


#Creamos una instancia de la cadena conversacional con el LLM y el objeto de memoria
conversation = ConversationChain(llm=llm,memory = memory,verbose=True)
#Ejemplo con RunnableWithMessageHistory: https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html


# In[ ]:


conversation.predict(input="Hola, ¿cómo estás?")


# In[ ]:


conversation.predict(input="Necesito un consejo para tener un gran día")


# In[ ]:


print(memory.buffer) #k limita el número de interacciones

