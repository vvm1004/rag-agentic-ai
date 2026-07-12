from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_gemini import get_gemini_llm


llm = get_gemini_llm(temperature=0.1)

prompt = PromptTemplate.from_template("""
Classify the emotion in the sentence.

Examples:

Sentence: I just won my first marathon!
Emotion: Joy

Sentence: I can't believe I lost my keys again.
Emotion: Frustration

Sentence: My best friend is moving to another country.
Emotion: Sadness

Sentence: I finally finished the project after weeks of hard work.
Emotion: Relief

Now classify this sentence:

Sentence: {sentence}
Emotion:
""")

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "sentence": "That movie was so scary I had to cover my eyes."
})

print(result)