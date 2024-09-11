#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías de templates

# In[ ]:


from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


# ## Generar plantillas de prompts

# In[ ]:


#Creamos la plantilla del sistema (system_template)
system_template="Eres una IA especializada en coches de tipo {tipo_coches} y generar artículos que se leen en {tiempo_lectura}."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)


# In[ ]:


system_message_prompt.input_variables


# In[ ]:


#Creamos la plantilla de usuario (human_template)
human_template="Necesito un artículo para vehículos con motor {peticion_tipo_motor}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


# In[ ]:


human_message_prompt.input_variables


# In[ ]:


#Creamos una plantilla de chat con la concatenación tanto de mensajes del sistema como del humano
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


# In[ ]:


chat_prompt.input_variables


# In[ ]:


# Completar el chat gracias al formateo de los mensajes
chat_prompt.format_prompt(peticion_tipo_motor="híbrido enchufable", tiempo_lectura="10 min", tipo_coches="japoneses")


# In[ ]:


#Transformamos el objeto prompt a una lista de mensajes y lo guardamos en "solicitud_completa" que es lo que pasaremos al LLM finalmente
solicitud_completa = chat_prompt.format_prompt(peticion_tipo_motor="híbrido enchufable", tiempo_lectura="10 min", tipo_coches="japoneses").to_messages()


# ## Obtener el resultado de la respuesta formateada

# In[ ]:


from langchain_openai import ChatOpenAI


# In[ ]:


f = open('../OpenAI_key.txt')
api_key = f.read()
chat = ChatOpenAI(openai_api_key=api_key)


# In[ ]:


result = chat.invoke(solicitud_completa)


# In[ ]:


result


# In[ ]:


print(result.content)


# In[ ]:




