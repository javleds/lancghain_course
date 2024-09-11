#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## 0.Importar librerías iniciales e instancia de modelo de chat

# In[ ]:


from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.agents import load_tools,initialize_agent,AgentType,create_react_agent,AgentExecutor
f = open('../OpenAI_key.txt')
api_key = f.read()
llm = ChatOpenAI(openai_api_key=api_key,temperature=0) #Recomendable temperatura a 0 para que el LLM no sea muy creativo, vamos a tener muchas herramientas a nuestra disposición y queremos que sea más determinista


# ## 1.Conectamos a la BBDD SQL

# In[ ]:


import mysql.connector #pip install mysql-connector-python


# In[ ]:


f = open('../password_sql.txt')
pass_sql = f.read()
# Configuración de la conexión a la base de datos
config = {
    'user': 'root',       
    'password': pass_sql, 
    'host': '127.0.0.1',         
    'database': 'world'          
}


# In[ ]:


# Conectar a la base de datos
conn = mysql.connector.connect(**config)
cursor = conn.cursor()


# # 2. Ejecutamos consulta manualmente (sin agentes Langchain)

# In[ ]:


# Definir la consulta manualmente: tengo una base de datos mysql en mi computadora local denominada "world" y una tabla "Country" 
#sobre la que quiero hacer la suma de la población en la columna "Population" para el continente Asia (columna "Continent")
query = """
    SELECT SUM(Population)
    FROM Country
    WHERE Continent = 'Asia';
    """

# Ejecutar la consulta
cursor.execute(query)
result = cursor.fetchone()


# In[ ]:


suma_poblacion = result[0] if result[0] is not None else 0
print(f"La suma de la población del continente Asia es: {suma_poblacion}")


# ## 3.Creamos el agente SQL 

# In[ ]:


from langchain_community.agent_toolkits import create_sql_agent
from langchain.sql_database import SQLDatabase


# In[ ]:


# Crear una cadena de conexión a la base de datos MySQL
connection_string = f"mysql+mysqlconnector://{config['user']}:{config['password']}@{config['host']}/{config['database']}"

# Crear una instancia de la base de datos SQL
db = SQLDatabase.from_uri(connection_string)


# In[ ]:


agent = create_sql_agent(
    llm,
    db=db,
    verbose=True
)


# In[ ]:


agent.invoke("Dime la población total de Asia")


# In[ ]:


result = agent.invoke("Dime el promedio de la esperanza de vida por cada una de las regiones ordenadas de mayor a menor")


# In[ ]:


# Mostrar el resultado
print(result["output"])


# In[ ]:


# Para utilizar few-shoots para las consultas SQL: https://python.langchain.com/v0.1/docs/use_cases/sql/agents/

