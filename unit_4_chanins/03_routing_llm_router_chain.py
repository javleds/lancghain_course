#!/usr/bin/env python
# coding: utf-8

# <em style="text-align:center">Copyright Iván Pinar Domínguez</em>

# #  Enrutamiento con LLM Router Chain

# ## Importar librerías iniciales e instancia de modelo de chat

# In[ ]:


from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import SequentialChain, LLMChain #importamos el SequentialChain que es el modelo completo
f = open('../OpenAI_key.txt')
api_key = f.read()
llm = ChatOpenAI(openai_api_key=api_key)


# ### Plantillas de enrutamiento

# In[ ]:


#Template (prompt) para soporte básico a clientes de coches
plantilla_soporte_basico_cliente = '''Eres una persona que asiste a los clientes de automóviles con preguntas básicas que pueden
necesitar en su día a día y que explica los conceptos de una manera que sea simple de entender. Asume que no tienen conocimiento
previo. Esta es la pregunta del usuario/n{input}'''


# In[ ]:


#Template (prompt) para soporte avanzados a nuestros expertos en mecánica
plantilla_soporte_avanzado_mecánico = '''Eres un experto en mecánica que explicas consultas avanzadas a los mecánicos
de la plantilla. Puedes asumir que cualquier que está preguntando tiene conocimientos avanzados de mecánica. 
Esta es la pregunta del usuario/n{input}'''


# In[ ]:


#Se pueden añadir todas las plantillas que necesitemos


# ### Prompts a enrutar

# In[ ]:


#Debemos crear una lista de diccionarios, cada diccionario contiene su nombre, la descripción (en base a la cual el enrutador
#hará su trabajo) y el prompt a usar en cada caso
prompt_infos = [
    {'name':'mecánica básica','description': 'Responde preguntas básicas de mecánicas a clientes',
     'prompt_template':plantilla_soporte_basico_cliente},
    {'name':'mecánica avanzada','description': 'Responde preguntas avanzadas de mecánica a expertos con conocimiento previo',
     'prompt_template':plantilla_soporte_avanzado_mecánico},
    
]


# In[ ]:


#TODO LO QUE VIENE A CONTINUACIÓN ES AGNÓSTICO DEL CASO DE USO Y LO PUEDES UTILIZAR PARA TODOS LOS CASOS DE USO


# ### ConversationChain

# In[ ]:


from langchain.chains import LLMChain


# In[ ]:


#Creamos un diccionario de objetos LLMChain con las posibles cadenas destino
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain


# In[ ]:


destination_chains


# In[ ]:


#Creamos el prompt y cadena por defecto puesto que son argumento obligatorios que usaremos posteriormente
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm,prompt=default_prompt)


# ### Multi Routing Template

# In[ ]:


#Importamos una plantilla que podremos formatear su parámetro {destinations} que tendrá cada nombre y descripción de la información de prompts
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE


# In[ ]:


print(MULTI_PROMPT_ROUTER_TEMPLATE) #El parámetro importante es {destinations}, debemos formatearlo en tipo string


# ### Destinos de Routing

# In[ ]:


#Creamos una string global con todos los destinos de routing usando el nombre y descripción de "prompt_infos"
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)


# In[ ]:


destinations_str


# ### Router Prompt

# In[ ]:


from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser


# In[ ]:


router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str #Formateamos la plantilla con nuestros destinos en la string destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(), #Para transformar el objeto JSON parseándolo a una string
)


# In[ ]:


print(router_template) #verificamos que se ha formateado correctamente


# ### Routing Chain Call

# In[ ]:


from langchain.chains.router import MultiPromptChain


# In[ ]:


router_chain = LLMRouterChain.from_llm(llm, router_prompt)


# In[ ]:


chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, #El objeto con los posibles LLMChain que creamos al inicio
                         default_chain=default_chain, verbose=True #Indicamos el LLMChain por defecto (obligatorio)
                        )


# In[ ]:


chain.invoke("¿Cómo cambio el aceite de mi coche?")


# In[ ]:


chain.invoke("¿Cómo funciona internamente un catalizador?")


# In[ ]:




