#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()

# Creamos la plantilla del sistema (system_template)
system_template = "Eres una IA especializada en coches de tipo {tipo_coches} y generar artículos que se leen en {" \
                  "tiempo_lectura}."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Creamos la plantilla de usuario (human_template)
human_template = "Necesito un artículo para vehículos con motor {peticion_tipo_motor}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Creamos una plantilla de chat con la concatenación tanto de mensajes del sistema como del humano
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Completar el chat gracias al formateo de los mensajes
chat_prompt.format_prompt(peticion_tipo_motor="híbrido enchufable", tiempo_lectura="10 min", tipo_coches="japoneses")

# Transformamos el objeto prompt a una lista de mensajes y lo guardamos en "solicitud_completa" que es lo que
# pasaremos al LLM finalmente
solicitud_completa = chat_prompt.format_prompt(peticion_tipo_motor="híbrido enchufable", tiempo_lectura="10 min",
                                               tipo_coches="japoneses").to_messages()

# ## Obtener el resultado de la respuesta formateada
chat = ChatOpenAI()
result = chat.invoke(solicitud_completa)
print(result.content)
