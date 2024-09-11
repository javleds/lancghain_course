#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías iniciales e instancia de modelo de chat

# In[1]:


from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.agents import load_tools,initialize_agent,AgentType,create_react_agent,AgentExecutor
f = open('../OpenAI_key.txt')
api_key = f.read()
llm = ChatOpenAI(openai_api_key=api_key,temperature=0) #Recomendable temperatura a 0 para que el LLM no sea muy creativo, vamos a tener muchas herramientas a nuestra disposición y queremos que sea más determinista


# # CASO DE USO 1

# In[2]:


from langchain.memory import ConversationBufferMemory


# In[3]:


memory = ConversationBufferMemory(memory_key="chat_history") #ponemos una denominada clave a la memoria "chat_history"


# In[4]:


tools = load_tools(["wikipedia","llm-math",],llm=llm)


# In[6]:


agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,memory=memory,verbose=True)


# In[7]:


agent.invoke("Dime 5 productos esenciales para el mantenimiento del vehículo.")


# In[8]:


agent.invoke("¿Cuál de los anteriores es el más importante?")


# In[9]:


agent.invoke("Necesito la respuesta anterior en castellano")

