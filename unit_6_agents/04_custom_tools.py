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


# # CASO DE USO 1

# ## Creamos nuestra herramienta personalizada 

# In[ ]:


from langchain.agents import tool


# In[ ]:


@tool
def persona_amable (text: str) -> str:
    '''Retorna la persona más amable. Se espera que la entrada esté vacía "" 
    y retorna la persona más amable del universo'''
    return "Miguel Celebres"


# ## Definimos las herramientas a las que tendrá acceso el agente y ejecutamos

# ### Ejemplo sin función personalizada 

# In[ ]:


tools = load_tools(["wikipedia","llm-math",],llm=llm)


# In[ ]:


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)


# In[ ]:


agent.invoke("¿Quién es la persona más amable del universo?")


# ### Ejemplo con función personalizada 

# In[ ]:


tools = tools + [persona_amable]


# In[ ]:


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)


# In[ ]:


agent.invoke("¿Quién es la persona más amable del universo?")


# # CASO DE USO 2: API interna

# In[ ]:


@tool
def nombre_api_interna(text: str) -> str:
    '''Conecta a la API_xx que realiza la tarea xx, debes usar esta API Key'''
    ##Definir conexión a la API interna y devolver un resultado
    return resultado


# # CASO DE USO 3: Consultar hora actual

# In[ ]:


# Solicitud con las herramientas actuales no proporciona el resultado que queremos
agent.invoke("¿Cuál es la hora actual?")


# ## Creamos nuestra función personalizada 

# In[ ]:


from datetime import datetime


# In[ ]:


@tool
def hora_actual(text: str)->str:
    '''Retorna la hora actual, debes usar esta función para cualquier consulta sobre la hora actual. Para fechas que no sean
    la hora actual, debes usar otra herramienta. La entrada está vacía y la salida retorna una string'''
    return str(datetime.now())


# In[ ]:


tools = tools + [hora_actual]


# In[ ]:


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)


# In[ ]:


# Solicitud con las herramientas actuales SÍ proporciona el resultado que queremos
agent.invoke("¿Cuál es la hora actual?")

