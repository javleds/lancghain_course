#!/usr/bin/env python
# coding: utf-8

# pip install numexpr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import (
    load_tools,
    initialize_agent,
    AgentType,
    create_react_agent,
    AgentExecutor
)

load_dotenv()
llm = ChatOpenAI(temperature=0)

# Lista de herramientas disponibles: https://python.langchain.com/v0.1/docs/integrations/tools/
tools = load_tools(["llm-math"], llm=llm)

# Usamos el Zero Shot porque no estamos dando ningún ejemplo
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False, handle_parsing_errors=True)
response1 = agent.run("Dime cuánto es 1598 multiplicado por 1983 y después sumas 1000")
print('\nPrimera respuesta:')
print(response1)

template = '''Responde lo mejor que puedas usando tu conocimiento como LLM o bien las siguientes herramientas:
{tools}
Utiliza el siguiente formato:
Pregunta: la pregunta de entrada que debes responder
Pensamiento: siempre debes pensar en qué hacer
Acción: la acción a realizar debe ser una de [{tool_names}]
Entrada de acción: la entrada a la acción.
Observación: el resultado de la acción.
... (este Pensamiento/Acción/Introducción de Acción/Observación puede repetirse N veces,si no consigues el resultado tras 5 intentos, para la ejecución)
Pensamiento: ahora sé la respuesta final
Respuesta final: la respuesta final a la pregunta de entrada original
¡Comenzar! Recuerda que no siempre es necesario usar las herramientas
Pregunta: {input}
Pensamiento:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)
agente = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agente,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True
)

respuesta = agent_executor.invoke({"input": "Dime cuánto es 1598 multiplicado por 1983"})
print('\nSegunda respuesta con tools respuesta:')
print(respuesta)
