#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain, LLMChain

load_dotenv()
llm = ChatOpenAI()

template1 = "Dame un resumen del rendimiento de este trabajador:\n{revision_rendimiento}"
prompt1 = ChatPromptTemplate.from_template(template1)
chain_1 = LLMChain(llm=llm, prompt=prompt1, output_key="resumen_revision")

template2 = "Identifica las debilidades de este trabajador dentro de de este resumen de la revisión:\n{resumen_revision}"
prompt2 = ChatPromptTemplate.from_template(template2)
chain_2 = LLMChain(llm=llm, prompt=prompt2, output_key="debilidades")

template3 = "Crea un plan de mejora para ayudar en estas debilidades:\n{debilidades}"
prompt3 = ChatPromptTemplate.from_template(template3)
chain_3 = LLMChain(llm=llm, prompt=prompt3, output_key="plan_mejora")

# Change verbose=False to more details
seq_chain = SequentialChain(chains=[chain_1, chain_2, chain_3],
                            input_variables=['revision_rendimiento'],
                            output_variables=['resumen_revision', 'debilidades', 'plan_mejora'],
                            verbose=False)

revision_rendimiento_empleado = '''
Revisión de Rendimiento del Empleado

Nombre del Empleado: Juan Pérez
Posición: Analista de Datos
Período Evaluado: Enero 2023 - Junio 2023

Fortalezas:
Juan ha demostrado un fuerte dominio de las herramientas analíticas y ha proporcionado informes detallados y precisos que han sido de gran ayuda para la toma de decisiones estratégicas. Su capacidad para trabajar en equipo y su disposición para ayudar a los demás también han sido notables. Además, ha mostrado una gran ética de trabajo y una actitud positiva en el entorno laboral.

Debilidades:
A pesar de sus muchas fortalezas, Juan ha mostrado áreas que necesitan mejoras. En particular, se ha observado que a veces tiene dificultades para manejar múltiples tareas simultáneamente, lo que resulta en retrasos en la entrega de proyectos. También ha habido ocasiones en las que la calidad del trabajo ha disminuido bajo presión. Además, se ha identificado una necesidad de mejorar sus habilidades de comunicación, especialmente en lo que respecta a la presentación de datos complejos de manera clara y concisa a los miembros no técnicos del equipo. Finalmente, se ha notado una falta de proactividad en la búsqueda de soluciones a problemas imprevistos, confiando a menudo en la orientación de sus superiores en lugar de tomar la iniciativa.
'''

results = seq_chain.invoke(revision_rendimiento_empleado)

print('\nPlan de mejora:')
print(results['plan_mejora'])
print('\nDebilidades:')
print(results["debilidades"])
