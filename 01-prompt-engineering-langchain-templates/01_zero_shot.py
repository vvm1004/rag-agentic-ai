from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_gemini import get_gemini_llm


llm = get_gemini_llm(temperature=0.1)

prompt = PromptTemplate.from_template("""
Classify the following movie review as Positive or Negative.

Review:
{review}

Return only one word: Positive or Negative.
""")

chain = prompt | llm | StrOutputParser()

review = """
The movie was boring, predictable, and way too long.
The acting felt unnatural and I would not recommend it.
"""

result = chain.invoke({
    "review": review
})

print(result)