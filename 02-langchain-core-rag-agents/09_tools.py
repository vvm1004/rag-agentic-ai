from tools_lib import (
    calculator,
    format_text,
)


def display_tool_metadata() -> None:
    """Display the metadata available to an agent."""
    print("=== CALCULATOR TOOL ===")
    print("Name:", calculator.name)
    print("Description:", calculator.description)
    print("Input schema:", calculator.args)

    print("\n=== FORMAT TEXT TOOL ===")
    print("Name:", format_text.name)
    print("Description:", format_text.description)
    print("Input schema:", format_text.args)


def invoke_tools_directly() -> None:
    """Invoke tools without using an agent."""
    calculation_result = calculator.invoke(
        {
            "expression": "(25 + 63) * 4",
        }
    )

    formatted_text = format_text.invoke(
        {
            "mode": "titlecase",
            "text": (
                "langchain makes language model "
                "applications composable"
            ),
        }
    )

    print("\n=== DIRECT TOOL INVOCATION ===")
    print(
        "Calculation result:",
        calculation_result,
    )

    print(
        "Formatted text:",
        formatted_text,
    )


def main() -> None:
    display_tool_metadata()
    invoke_tools_directly()


if __name__ == "__main__":
    main()