#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

chat = ChatOpenAI()
resultado_simple = chat.invoke([HumanMessage(content="¿Puedes decirme dónde se encuentra Cáceres?")])

# Especificamos el SystemMessage para definir la personalidad que debe tomar el sistema
resultado_personalizado = chat.invoke(
    [SystemMessage(content='Eres un historiador que conoce los detalles de todas las ciudades del mundo'),
     HumanMessage(content='¿Puedes decirme dónde se encuentra Cáceres')])

print(resultado_personalizado.generations[0][0].text)

resultado = chat.generate(
    [
        [SystemMessage(content='Eres un historiador que conoce los detalles de todas las ciudades del mundo'),
         HumanMessage(content='¿Puedes decirme dónde se encuentra Cáceres')],
        [SystemMessage(content='Eres un joven rudo que no le gusta que le pregunten, solo quiere estar de fiesta'),
         HumanMessage(content='¿Puedes decirme dónde se encuentra Cáceres')]
    ]
)

# Resultado como historiador
print(resultado.generations[0][0].text)

# Resultado como joven rudo
print(resultado.generations[1][0].text)
