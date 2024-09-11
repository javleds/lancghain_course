#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# ## Importar librerías iniciales e instancia de modelo de chat

# In[ ]:


from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import SimpleSequentialChain, LLMChain,TransformChain
f = open('../OpenAI_key.txt')
api_key = f.read()
llm = ChatOpenAI(openai_api_key=api_key)


# ## Importamos documentos

# In[ ]:


from langchain.document_loaders import WikipediaLoader


# In[ ]:


consulta_wikipedia = input()


# In[ ]:


idioma_final = input()


# In[ ]:


loader = WikipediaLoader(query=consulta_wikipedia,lang="es",load_max_docs=10)


# In[ ]:


data = loader.load()


# In[ ]:


data[0].page_content


# In[ ]:


texto_entrada = data[0].page_content


# # TransformChain

# ### Definir la función de transformación personalizada

# In[ ]:


def transformer_function(inputs: dict) -> dict: #Toma de entrada un diccionario y lo devuelve con la transformación oportuna
    texto = inputs['texto']
    primer_parrafo = texto.split('\n')[0]
    return {'salida':primer_parrafo}


# In[ ]:


transform_chain = TransformChain(input_variables=['texto'],
                                 output_variables=['salida'],
                                 transform=transformer_function)


# ## Definir la cadena secuencial

# In[ ]:


#Creamos bloque LLMChain para resumir
template1 = "Crea un resumen en una línea del siguiente texto:\n{texto}"
prompt = ChatPromptTemplate.from_template(template1)
summary_chain = LLMChain(llm=llm,
                     prompt=prompt,
                     output_key="texto_resumen")


# In[ ]:


#Creamos bloque LLMChain para traducir
template2 = "Traduce a"+ idioma_final + "el siguiente texto:\n{texto}"
prompt = ChatPromptTemplate.from_template(template2)
#prompt.format_prompt(idioma=idioma_final)
translate_chain = LLMChain(llm=llm,
                     prompt=prompt,
                     output_key="texto_traducido")


# In[ ]:


sequential_chain = SimpleSequentialChain(chains=[transform_chain,summary_chain,translate_chain],
                                        verbose=True)


# In[ ]:


result = sequential_chain(texto_entrada)

