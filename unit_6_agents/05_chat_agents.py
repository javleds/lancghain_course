#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

load_dotenv()
llm = ChatOpenAI(temperature=0)

# ponemos una denominada clave a la memoria "chat_history"
memory = ConversationBufferMemory(memory_key="chat_history")
tools = load_tools(["wikipedia", "llm-math", ], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)

print('\n5 Productos mantenimiento vehiculo:')
print(agent.invoke({'input': 'Dime 5 productos esenciales para el mantenimiento del vehículo.'}))

print('\nEl más importante:')
print(agent.invoke({'input': '¿Cuál de los anteriores es el más importante?'}))

print('\nEn inglés:')
print(agent.invoke({'input': 'Necesito la respuesta anterior en inglés'}))
