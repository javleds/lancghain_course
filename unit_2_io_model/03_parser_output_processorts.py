#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.output_parsers import (
    CommaSeparatedListOutputParser,
    DatetimeOutputParser,
    OutputFixingParser
)

load_dotenv()

chat = ChatOpenAI()

# ## Parsear una lista de elementos separados por coma
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

# Respuesta imaginaria
respuesta = "coche, árbol, carretera"
print(output_parser.parse(respuesta))

# Creamos la plantilla de usuario (human_template) con la concatenación de la variable "request" (la solicitud) y la
# variable "format_instructions" con las instrucciones adicionales que le pasaremos al LLM
human_template = '{request}\n{format_instructions}'
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Creamos el prompt y le damos formato a las variables
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
# Transformamos el objeto prompt a una lista de mensajes y lo guardamos en "solicitud_completa" que es lo que
# pasaremos al LLM finalmente
solicitud_completa = chat_prompt.format_prompt(request="dime 5 características de los coches americanos",
                                               format_instructions=output_parser.get_format_instructions()).to_messages()

result = chat.invoke(solicitud_completa)
print(result.content)

output_parser.parse(result.content)

output_parser = DatetimeOutputParser()
print(output_parser.get_format_instructions())

template_text = "{request}\n{format_instructions}"
human_prompt = HumanMessagePromptTemplate.from_template(template_text)

chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
print(chat_prompt.format(request="¿Cuándo es el día de la declaración de independencia de los EEUU?",
                         format_instructions=output_parser.get_format_instructions()
                         ))

solicitud_completa = chat_prompt.format_prompt(
    request="¿Cuándo es el día de la declaración de independencia de los EEUU?",
    format_instructions=output_parser.get_format_instructions()
    ).to_messages()

result = chat.invoke(solicitud_completa)
print(result.content)
output_parser.parse(result.content)

# # Métodos para solucionar problemas de parseo
# ## Auto-Fix Parser

output_parser_dates = DatetimeOutputParser()
bad_formatted = result.content = result.content
print(bad_formatted)
new_parser = OutputFixingParser.from_llm(parser=output_parser_dates, llm=chat)
print(new_parser.parse(bad_formatted))

# ## Solucionar con System Prompt:
system_prompt = SystemMessagePromptTemplate.from_template("Tienes que responder únicamente con un patrón de fechas")
template_text = "{request}\n{format_instructions}"
human_prompt = HumanMessagePromptTemplate.from_template(template_text)

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

print(chat_prompt.format(request="¿Cuándo es el día de la declaración de independencia de los EEUU?",
                         format_instructions=output_parser_dates.get_format_instructions()
                         ))

solicitud_completa = chat_prompt.format_prompt(
    request="¿Cuándo es el día de la declaración de independencia de los EEUU?",
    format_instructions=output_parser_dates.get_format_instructions()
    ).to_messages()

result = chat.invoke(solicitud_completa)

print(result.content)
print(output_parser_dates.parse(result.content))
