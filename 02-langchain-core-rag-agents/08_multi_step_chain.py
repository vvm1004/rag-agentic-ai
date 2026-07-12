from langchain_core.output_parsers import (
    StrOutputParser,
)
from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_core.runnables import (
    RunnablePassthrough,
)

from common import get_chat_model


def main() -> None:
    model = get_chat_model(temperature=0.2)
    output_parser = StrOutputParser()

    sentiment_prompt = (
        PromptTemplate.from_template(
            """
Classify the sentiment of the product review.

Return exactly one value:
positive, neutral, or negative.

Review:
{review}
""".strip()
        )
    )

    summary_prompt = (
        PromptTemplate.from_template(
            """
Summarize the product review into three concise bullet points.

Review:
{review}

Detected sentiment:
{sentiment}
""".strip()
        )
    )

    response_prompt = (
        PromptTemplate.from_template(
            """
Write a helpful customer service response.

Rules:
- Thank the customer when the sentiment is positive.
- Acknowledge the problem when the sentiment is negative.
- Suggest a practical next step.
- Refer to specific details from the review.
- Use no more than 100 words.

Original review:
{review}

Sentiment:
{sentiment}

Summary:
{summary}
""".strip()
        )
    )

    sentiment_chain = (
        sentiment_prompt
        | model
        | output_parser
    )

    summary_chain = (
        summary_prompt
        | model
        | output_parser
    )

    response_chain = (
        response_prompt
        | model
        | output_parser
    )

    workflow = (
        RunnablePassthrough.assign(
            sentiment=sentiment_chain
        )
        | RunnablePassthrough.assign(
            summary=summary_chain
        )
        | RunnablePassthrough.assign(
            response=response_chain
        )
    )

    review = (
        "I am disappointed with this laptop. "
        "It overheats after thirty minutes, "
        "the battery lasts only three hours, "
        "and several keyboard keys have started sticking."
    )

    result = workflow.invoke(
        {
            "review": review,
        }
    )

    print("=== ORIGINAL REVIEW ===")
    print(result["review"])

    print("\n=== SENTIMENT ===")
    print(result["sentiment"])

    print("\n=== SUMMARY ===")
    print(result["summary"])

    print("\n=== CUSTOMER RESPONSE ===")
    print(result["response"])


if __name__ == "__main__":
    main()