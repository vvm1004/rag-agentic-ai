from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_gemini import get_gemini_llm


llm = get_gemini_llm(temperature=0.2)

prompt = PromptTemplate.from_template("""
Here is one example of explaining a technical concept simply:

Technical Concept: Blockchain
Simple Explanation: A blockchain is like a shared digital notebook. Many people have a copy of it, and when new information is added, everyone can check that it is valid.

Now explain this concept in the same simple style:

Technical Concept: {concept}
Simple Explanation:
""")

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "concept": "Machine Learning"
})

print(result)