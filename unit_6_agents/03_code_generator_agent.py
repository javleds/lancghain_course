#!/usr/bin/env python
# coding: utf-8

# pip install pandas seaborn
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool

load_dotenv()
llm = ChatOpenAI(temperature=0)

# ## Creamos el agente para crear y ejecutar código Python
agent = create_python_agent(tool=PythonREPLTool(),
                            llm=llm,
                            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

lista_ejemplo = [3, 1, 5, 3, 5, 6, 7, 3, 5, 10]
agent.invoke({'input': f'''ordena la lista {lista_ejemplo}'''})

df = pd.read_excel('datos_ventas_small.xlsx')

print(df.head())
print(agent.invoke({
    'input': f'''¿Qué sentencias de código tendría que ejecutar para obtener la suma de venta total agregada por Línea de Producto? Este sería el dataframe {df}, no tienes que ejecutar la sentencia, solo pasarme el código a ejecutar''',
}))

df.groupby('Línea Producto')['Venta total'].sum()

print(agent.invoke({
    'input': f'''¿Cuál es la suma agregada de la venta total para la línea de proudcto "Motorcycles"? Este sería el dataframe {df}'''
}))

print(agent.invoke({
    'input': f'''¿Qué sentencias de código tendría que ejecutar para tener una visualización con la librería Seaborn que agregue a nivel de Línea de Producto el total de venta? Este sería el dataframe {df}, recuerda que no tienes que ejecutar la sentencia, solo pasarme el código a ejecutar'''
}))

sns.barplot(x='Línea Producto', y='Venta total', data=df, estimator=sum)
