#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# #  ChatMessageHistory

# In[ ]:


from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
f = open('../OpenAI_key.txt')
api_key = f.read()
chat = ChatOpenAI(openai_api_key=api_key)


# In[ ]:


#Definimos el objeto de histórico de mensajes
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()


# In[ ]:


consulta = "Hola, ¿cómo estás? Necesito ayudar para reconfigurar el router"


# In[ ]:


#Vamos guardando en el objeto "history" los mensajes de usuario y los mensajes AI que queramos
history.add_user_message(consulta)


# In[ ]:


resultado = chat.invoke([HumanMessage(content=consulta)])


# In[ ]:


history.add_ai_message(resultado.content)


# In[ ]:


history


# In[ ]:


history.messages

