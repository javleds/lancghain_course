#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías e instancia de modelo de chat

# In[ ]:


from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
f = open('../OpenAI_key.txt')
api_key = f.read()
chat = ChatOpenAI(openai_api_key=api_key)


# ## Parsear una lista de elementos separados por coma

# In[ ]:


from langchain.output_parsers import CommaSeparatedListOutputParser


# In[ ]:


output_parser = CommaSeparatedListOutputParser()


# In[ ]:


format_instructions = output_parser.get_format_instructions() #Nos devuelve las instrucciones que va a pasar al LLM en función del parseador concreto


# In[ ]:


print(format_instructions)


# In[ ]:


#Respuesta imaginaria
respuesta = "coche, árbol, carretera"
output_parser.parse(respuesta)


# In[ ]:


#Creamos la plantilla de usuario (human_template) con la concatenación de la variable "request" (la solicitud) y la variable "format_instructions" con 
#las instrucciones adicionales que le pasaremos al LLM
human_template = '{request}\n{format_instructions}'
human_prompt = HumanMessagePromptTemplate.from_template(human_template)


# In[ ]:


#Creamos el prompt y le damos formato a las variables
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

chat_prompt.format_prompt(request="dime 5 características de los coches americanos",
                   format_instructions = output_parser.get_format_instructions()) #Las instrucciones son las que proporciona el propio parseador


# In[ ]:


#Transformamos el objeto prompt a una lista de mensajes y lo guardamos en "solicitud_completa" que es lo que pasaremos al LLM finalmente
solicitud_completa = chat_prompt.format_prompt(request="dime 5 características de los coches americanos",
                   format_instructions = output_parser.get_format_instructions()).to_messages()


# In[ ]:


result = chat.invoke(solicitud_completa)


# In[ ]:


result.content


# In[ ]:


# Convertir a la salida esperada
output_parser.parse(result.content)


# ## Parsear formatos de fecha

# In[ ]:


from langchain.output_parsers import DatetimeOutputParser


# In[ ]:


output_parser = DatetimeOutputParser()


# In[ ]:


print(output_parser.get_format_instructions())


# In[ ]:


template_text = "{request}\n{format_instructions}"
human_prompt=HumanMessagePromptTemplate.from_template(template_text)


# In[ ]:


chat_prompt = ChatPromptTemplate.from_messages([human_prompt])


# In[ ]:


print(chat_prompt.format(request="¿Cuándo es el día de la declaración de independencia de los EEUU?",
                   format_instructions=output_parser.get_format_instructions()
                   ))


# In[ ]:


solicitud_completa = chat_prompt.format_prompt(request="¿Cuándo es el día de la declaración de independencia de los EEUU?",
                   format_instructions=output_parser.get_format_instructions()
                   ).to_messages()


# In[ ]:


result = chat.invoke(solicitud_completa)


# In[ ]:


result.content


# In[ ]:


output_parser.parse(result.content)


# 
# # Métodos para solucionar problemas de parseo
# 
# ## Auto-Fix Parser

# In[ ]:


from langchain.output_parsers import OutputFixingParser

output_parser_dates = DatetimeOutputParser()

misformatted = result.content


# In[ ]:


misformatted


# In[ ]:


new_parser = OutputFixingParser.from_llm(parser=output_parser_dates, llm=chat)


# In[ ]:


new_parser.parse(misformatted)


# ## Solucionar con System Prompt:

# In[ ]:


system_prompt = SystemMessagePromptTemplate.from_template("Tienes que responder únicamente con un patrón de fechas")
template_text = "{request}\n{format_instructions}"
human_prompt=HumanMessagePromptTemplate.from_template(template_text)


# In[ ]:


chat_prompt = ChatPromptTemplate.from_messages([system_prompt,human_prompt])


# In[ ]:


print(chat_prompt.format(request="¿Cuándo es el día de la declaración de independencia de los EEUU?",
                   format_instructions=output_parser_dates.get_format_instructions()
                   ))


# In[ ]:


solicitud_completa = chat_prompt.format_prompt(request="¿Cuándo es el día de la declaración de independencia de los EEUU?",
                   format_instructions=output_parser_dates.get_format_instructions()
                   ).to_messages()


# In[ ]:


result = chat.invoke(solicitud_completa)


# In[ ]:


result.content


# In[ ]:


output_parser_dates.parse(result.content)

