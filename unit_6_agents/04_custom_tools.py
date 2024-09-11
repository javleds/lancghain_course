#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import tool, load_tools, initialize_agent, AgentType

load_dotenv()
llm = ChatOpenAI(temperature=0)


# CASO DE USO 1
@tool
def persona_amable(text: str) -> str:
    '''Retorna la persona más amable. Se espera que la entrada esté vacía "" 
    y retorna la persona más amable del universo'''
    return "Miguel Celebres"


@tool
def nombre_api_interna(text: str) -> str:
    """Conecta a la API_xx que realiza la tarea xx, debes usar esta API Key"""
    # Definir conexión a la API interna y devolver un resultado
    return 'El resultado de la API interna'


@tool
def hora_actual(text: str) -> str:
    """Retorna la hora actual, debes usar esta función para cualquier consulta sobre la hora actual. Para fechas que no sean
    la hora actual, debes usar otra herramienta. La entrada está vacía y la salida retorna una string"""
    return str(datetime.now())


tools = load_tools(["wikipedia", "llm-math", ], llm=llm)
tools = tools + [persona_amable, nombre_api_interna, hora_actual]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

print(agent.invoke({'input': '¿Quién es la persona más amable del universo?'}))
print(agent.invoke({'input': '¿Cuál es la hora actual?'}))
