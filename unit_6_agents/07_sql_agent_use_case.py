#!/usr/bin/env python
# coding: utf-8

# Para utilizar few-shoots para las consultas SQL:
# https://python.langchain.com/v0.1/docs/use_cases/sql/agents/

# pip install mysql-connector-python
import os
import mysql.connector
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase

load_dotenv()
llm = ChatOpenAI(temperature=0)

# Configuración de la conexión a la base de datos
config = {
    'user': os.environ.get('DATABASE_USER'),
    'password': os.environ.get('DATABASE_PASSWORD'),
    'host': os.environ.get('DATABASE_HOST'),
    'port': os.environ.get('DATABASE_PORT'),
    'database': os.environ.get('DATABASE_DATABASE'),
}

conn = mysql.connector.connect(**config)
cursor = conn.cursor()

query = """
    SELECT SUM(Population)
    FROM country
    WHERE Continent = 'Asia';
    """

cursor.execute(query)
result = cursor.fetchone()

suma_poblacion = result[0] if result[0] is not None else 0
print(f"La suma de la población del continente Asia es: {suma_poblacion}")


connection_string = os.environ.get('DATABASE_URL')
db = SQLDatabase.from_uri(connection_string)
agent = create_sql_agent(
    llm,
    db=db,
    verbose=False
)

print(agent.invoke({'input': 'Dime la suma de la población total de Asia'}))

result = agent.invoke({
    'input': 'Dime el promedio de la esperanza de vida por cada una de las regiones ordenadas de mayor a menor'
})

print(result["output"])
