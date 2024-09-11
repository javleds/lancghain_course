#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    load_prompt,
    PromptTemplate
)

load_dotenv()
chat = ChatOpenAI()

# Guardar plantilla prompt
plantilla = "Pregunta: {pregunta_usuario}\n\nRespuesta: Vamos a verlo paso a paso."
prompt = PromptTemplate(template=plantilla)
prompt.save("prompt.json")

# ### Cargar plantilla prompt
prompt_cargado = load_prompt('prompt.json')

print(prompt_cargado)
