#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
chat = ChatOpenAI()

history = ChatMessageHistory()
consulta = "Hola, ¿cómo estás? Necesito ayudar para reconfigurar el router"
history.add_user_message(consulta)
resultado = chat.invoke([HumanMessage(content=consulta)])
history.add_ai_message(resultado.content)

print('\n History:')
print(history)

print('\n History messages:')
print(history.messages)
