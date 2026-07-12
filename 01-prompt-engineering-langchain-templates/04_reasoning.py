from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_gemini import get_gemini_llm


llm = get_gemini_llm(temperature=0.2)

prompt = PromptTemplate.from_template("""
Solve the following problem carefully.

Problem:
{problem}

Explain the key steps briefly, then give the final answer.
""")

chain = prompt | llm | StrOutputParser()

problem = """
A store had 22 apples.
They sold 15 apples today.
Then they received a new delivery of 8 apples.
How many apples are there now?
"""

result = chain.invoke({
    "problem": problem
})

print(result)