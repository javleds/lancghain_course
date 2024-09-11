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


# #  Integración Wikipedia

# In[ ]:


from langchain.document_loaders import WikipediaLoader # pip install wikipedia en una terminal


# In[ ]:


def responder_wikipedia(persona,pregunta_arg):
    # Obtener artículo de wikipedia
    docs = WikipediaLoader(query=persona,lang="es",load_max_docs=10) #parámetros posibles en: https://python.langchain.com/v0.2/docs/integrations/document_loaders/wikipedia/
    contexto_extra = docs.load()[0].page_content #para que sea más rápido solo pásamos el primer documento [0] como contexto extra
    
    # Pregunta de usuario
    human_prompt = HumanMessagePromptTemplate.from_template('Responde a esta pregunta\n{pregunta}, aquí tienes contenido extra:\n{contenido}')
    
    # Construir prompt
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    
    # Resultado
    result = chat.invoke(chat_prompt.format_prompt(pregunta=pregunta_arg,contenido=contexto_extra).to_messages())
    
    print(result.content)


# In[ ]:


responder_wikipedia("Fernando Alonso","¿Cuándo nació?")


# In[ ]:




