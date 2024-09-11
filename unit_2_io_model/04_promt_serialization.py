#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías e instancia de modelo de chat

# In[ ]:


from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
f = open('../OpenAI_key.txt')
api_key = f.read()
chat = ChatOpenAI(openai_api_key=api_key)


# ### Guardar plantilla prompt

# In[ ]:


plantilla = "Pregunta: {pregunta_usuario}\n\nRespuesta: Vamos a verlo paso a paso."
prompt = PromptTemplate(template=plantilla)
prompt.save("prompt.json")


# ### Cargar plantilla prompt

# In[ ]:


from langchain.prompts import load_prompt


# In[ ]:


prompt_cargado = load_prompt('prompt.json')


# In[ ]:


prompt_cargado

