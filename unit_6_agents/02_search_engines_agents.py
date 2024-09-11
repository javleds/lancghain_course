#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías iniciales e instancia de modelo de chat

# In[ ]:


from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.agents import load_tools,initialize_agent,AgentType,create_react_agent,AgentExecutor
f = open('../OpenAI_key.txt')
api_key = f.read()
llm = ChatOpenAI(openai_api_key=api_key,temperature=0) #Recomendable temperatura a 0 para que el LLM no sea muy creativo, vamos a tener muchas herramientas a nuestra disposición y queremos que sea más determinista


# In[ ]:


#Web de API que usa buscadores: https://serpapi.com/ --> Registrarse


# ## Definir SERP API Key

# In[ ]:


f = open('../SERP API Key.txt')
serp_api_key = f.read()


# In[ ]:


#Definir variable de entorno para que funcione correctamente:
import os
os.environ["SERPAPI_API_KEY"]=serp_api_key #Si no está definida el error nos dará el nombre de la variable de entorno que espera


# ## Definimos las herramientas a las que tendrá acceso el agente

# In[ ]:


tools = load_tools(["serpapi","llm-math",],llm=llm)


# In[ ]:


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)


# In[ ]:


agent.invoke("¿En qué año nació Einstein? ¿Cuál es el resultado de ese año multiplicado por 3?")


# In[ ]:




