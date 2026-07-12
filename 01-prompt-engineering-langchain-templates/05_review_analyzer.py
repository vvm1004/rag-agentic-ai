from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_gemini import get_gemini_llm


llm = get_gemini_llm(temperature=0.2)

prompt = PromptTemplate.from_template("""
Analyze the following product review:

"{review}"

Return the result in this exact format:
- Sentiment: positive, negative, or neutral
- Key Features Mentioned: list the product features mentioned
- Summary: one sentence only
""")

chain = prompt | llm | StrOutputParser()

reviews = [
    "I love this smartphone! The camera quality is exceptional and the battery lasts all day. The only downside is that it heats up a bit during gaming.",
    "This laptop is terrible. It's slow, crashes frequently, and the keyboard stopped working after just two months. Customer service was unhelpful.",
    "The headphones are okay. The sound quality is decent, but the build feels cheap and the ear pads are not very comfortable.",
]

for index, review in enumerate(reviews, start=1):
    print(f"\n==== Review #{index} ====")

    result = chain.invoke({
        "review": review
    })

    print(result)