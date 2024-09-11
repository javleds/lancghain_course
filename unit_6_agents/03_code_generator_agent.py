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


from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool


# ## Creamos el agente para crear y ejecutar código Python 

# In[ ]:


agent = create_python_agent(tool=PythonREPLTool(),
                           llm=llm,
                           agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)


# In[ ]:


lista_ejemplo = [3,1,5,3,5,6,7,3,5,10]


# In[ ]:


agent.invoke(f'''ordena la lista {lista_ejemplo}''')


# ## Ejemplo con un dataframe

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_excel('datos_ventas_small.xlsx')


# In[ ]:


df.head()


# In[ ]:


agent.invoke(f'''¿Qué sentencias de código tendría que ejecutar para obtener la suma de venta total agregada por Línea de Producto? Este sería el dataframe {df}, no tienes que ejecutar la sentencia, solo pasarme el código a ejecutar''')


# In[ ]:


df.groupby('Línea Producto')['Venta total'].sum()


# In[ ]:


agent.invoke(f'''¿Cuál es la suma agregada de la venta total para la línea de proudcto "Motorcycles"? Este sería el dataframe {df}''')


# In[ ]:


agent.invoke(f'''¿Qué sentencias de código tendría que ejecutar para tener una visualización con la librería Seaborn que agregue a nivel de Línea de Producto el total de venta? Este sería el dataframe {df}, recuerda que no tienes que ejecutar la sentencia, solo pasarme el código a ejecutar''')


# In[ ]:


import seaborn as sns
sns.barplot(x='Línea Producto', y='Venta total', data=df, estimator=sum)

