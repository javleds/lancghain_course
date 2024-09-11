#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.sequential import SimpleSequentialChain

load_dotenv()

llm = ChatOpenAI()
template = "Dame un simple resumen con un listado de puntos para un post de un blog acerca de {tema}"
prompt1 = ChatPromptTemplate.from_template(template)
chain_1 = LLMChain(llm=llm, prompt=prompt1)

template = "Escribe un post completo usando este resumen: {resumen}"
prompt2 = ChatPromptTemplate.from_template(template)
chain_2 = LLMChain(llm=llm, prompt=prompt2)

# verbose=True nos irá dando paso a paso lo que hace, permitiendo ver los resultados intermedios
full_chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=False)
result = full_chain.invoke({'input': "Inteligencia Artificial"})

print(result['output'])
