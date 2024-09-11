#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain

load_dotenv()
llm = ChatOpenAI()

# Template (prompt) para soporte básico a clientes de coches
plantilla_soporte_basico_cliente = '''Eres una persona que asiste a los clientes de automóviles con preguntas básicas que pueden
necesitar en su día a día y que explica los conceptos de una manera que sea simple de entender. Asume que no tienen conocimiento
previo. Esta es la pregunta del usuario/n{input}'''

# Template (prompt) para soporte avanzados a nuestros expertos en mecánica
plantilla_soporte_avanzado_mecanico = '''Eres un experto en mecánica que explicas consultas avanzadas a los mecánicos
de la plantilla. Puedes asumir que cualquier que está preguntando tiene conocimientos avanzados de mecánica. 
Esta es la pregunta del usuario/n{input}'''

prompt_infos = [
    {'name': 'mecánica básica', 'description': 'Responde preguntas básicas de mecánicas a clientes',
     'prompt_template': plantilla_soporte_basico_cliente},
    {'name': 'mecánica avanzada',
     'description': 'Responde preguntas avanzadas de mecánica a expertos con conocimiento previo',
     'prompt_template': plantilla_soporte_avanzado_mecanico},
]

# TODO LO QUE VIENE A CONTINUACIÓN ES AGNÓSTICO DEL CASO DE USO Y LO PUEDES UTILIZAR PARA TODOS LOS CASOS DE USO

# Creamos un diccionario de objetos LLMChain con las posibles cadenas destino
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

# Creamos el prompt y cadena por defecto puesto que son argumento obligatorios que usaremos posteriormente
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

print('\nMulti router template:')
print(MULTI_PROMPT_ROUTER_TEMPLATE)  # El parámetro importante es {destinations}, debemos formatearlo en tipo string

# ### Destinos de Routing
# Creamos una string global con todos los destinos de routing usando el nombre y descripción de "prompt_infos"
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

print('\nFormatted template:')
print(destinations_str)

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str  # Formateamos la plantilla con nuestros destinos en la string destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),  # Para transformar el objeto JSON parseándolo a una string
)

print('\nRouter template:')
print(router_template)


router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destination_chains,
                         default_chain=default_chain,
                         verbose=True)

simple_answer = chain.invoke({'input': '¿Cómo cambio el aceite de mi coche?'})
print('\nSimple answer:')
print(simple_answer)

print('\nAdvanced answer:')
advanced_answer = chain.invoke({'input': '¿Cómo funciona internamente un catalizador?'})
print(advanced_answer)
