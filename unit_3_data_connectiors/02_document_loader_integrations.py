#!/usr/bin/env python
# coding: utf-8

# pip install wikipedia en una terminal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders import WikipediaLoader

load_dotenv()
chat = ChatOpenAI()


def responder_wikipedia(persona: str, pregunta_arg: str) -> str:
    # Obtener artículo de wikipedia
    # parámetros posibles en: https://python.langchain.com/v0.2/docs/integrations/document_loaders/wikipedia/
    docs = WikipediaLoader(query=persona, lang="es",
                           load_max_docs=10)
    # para que sea más rápido solo pasamos el primer documento [0] como contexto extra
    contexto_extra = docs.load()[0].page_content

    # Pregunta de usuario
    human_prompt = HumanMessagePromptTemplate.from_template(
        'Responde a esta pregunta\n{pregunta}, aquí tienes contenido extra:\n{contenido}')

    # Construir prompt
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

    # Resultado
    result = chat.invoke(chat_prompt.format_prompt(pregunta=pregunta_arg, contenido=contexto_extra).to_messages())

    return str(result.content)


respuesta = responder_wikipedia("Fernando Alonso", "¿Cuándo nació?")

print(respuesta)
