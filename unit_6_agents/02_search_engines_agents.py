#!/usr/bin/env python
# coding: utf-8

# Web de API que usa buscadores: https://serpapi.com/ --> Registrarse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

load_dotenv()
llm = ChatOpenAI(temperature=0)

tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

response = agent.invoke({'input': '¿En qué año nació Einstein? ¿Cuál es el resultado de ese año multiplicado por 3?'})

print(response)