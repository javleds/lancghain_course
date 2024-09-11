#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías y clave OpenAI

# In[5]:


from dotenv import load_dotenv
load_dotenv()


# In[2]:


from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
#Alternativa para importar tipos de mensajes: from langchain_core.messages import HumanMessage, SystemMessage


# In[6]:


chat = ChatOpenAI()


# ## Obtener 1 resultado invocando al chat de OpenAI

# In[7]:


resultado = chat.invoke([HumanMessage(content="¿Puedes decirme dónde se encuentra Cáceres?")])


# In[8]:


resultado


# In[9]:


resultado.content


# In[ ]:


#Especificamos el SystemMessage para definir la personalidad que debe tomar el sistema


# In[ ]:


resultado = chat.invoke([SystemMessage(content='Eres un historiador que conoce los detalles de todas las ciudades del mundo'),
               HumanMessage(content='¿Puedes decirme dónde se encuentra Cáceres')])


# In[ ]:


resultado.content


# ## Obtener varios resultados invocando al chat de OpenAI con "generate"

# In[ ]:


resultado = chat.generate(
    [
        [SystemMessage(content='Eres un historiador que conoce los detalles de todas las ciudades del mundo'),
         HumanMessage(content='¿Puedes decirme dónde se encuentra Cáceres')],
        [SystemMessage(content='Eres un joven rudo que no le gusta que le pregunten, solo quiere estar de fiesta'),
         HumanMessage(content='¿Puedes decirme dónde se encuentra Cáceres')]
    ]
)


# In[ ]:


#Resultado con primer sistema
print(resultado.generations[0][0].text)


# In[ ]:


#Resultado con segundo sistema
print(resultado.generations[1][0].text)

