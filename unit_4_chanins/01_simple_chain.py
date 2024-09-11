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


# ##  Creación objeto LLMChain

# In[ ]:


human_message_prompt = HumanMessagePromptTemplate.from_template(
        "Dame un nombre de compañía que sea simpático para una compañía que fabrique {producto}"
    )


# In[ ]:


chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])


# In[ ]:


from langchain.chains import LLMChain
chain = LLMChain(llm=chat, prompt=chat_prompt_template)


# In[ ]:


print(chain.invoke(input="Lavadoras"))


# #  Cadena Secuencia Simple

# In[ ]:


from langchain.chains.sequential import SimpleSequentialChain


# In[ ]:


llm = ChatOpenAI(openai_api_key=api_key)


# In[ ]:


template = "Dame un simple resumen con un listado de puntos para un post de un blog acerca de {tema}"
prompt1 = ChatPromptTemplate.from_template(template)
chain_1 = LLMChain(llm=llm,prompt=prompt1)


# In[ ]:


template = "Escribe un post completo usando este resumen: {resumen}"
prompt2 = ChatPromptTemplate.from_template(template)
chain_2 = LLMChain(llm=llm,prompt=prompt2)


# In[ ]:


full_chain = SimpleSequentialChain(chains=[chain_1,chain_2],
                                  verbose=True) #verbose=True nos irá dando paso a paso lo que hace, pudiendo ver los resultados intermedios


# In[ ]:


result = full_chain.invoke(input="Inteligencia Artificial")


# In[ ]:


print(result['output'])

