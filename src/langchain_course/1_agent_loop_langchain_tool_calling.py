from typing import Dict

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10

MODEL = "qwen3:1.7b"


@tool
def get_product_price(product_name: str) -> float:
    """Look up the price of a product by its name."""
    print(f"Looking up price for product: {product_name}")
    prices: Dict[str, float] = {
        "laptop": 999.99,
        "phone": 599.99,
        "headphones": 199.99,
        "tablet": 399.99,
        "keyboard": 99.99,
    }
    return prices.get(product_name.lower(), -1.0)  # Return -1.0 if product not found


@tool
def apply_discount(price: float, disacount_tier: str) -> float:
    """Apply a disacount tier to a price and return the final price.
    Available tiers: bronze, silver, gold, platinum
    """
    print(
        f"    >> Executing apply_discount(price={price}, disacount_tier={disacount_tier})"
    )
    discount_rates: Dict[str, float] = {
        "bronze": 0.05,
        "silver": 0.10,
        "gold": 0.15,
        "platinum": 0.20,
    }
    discount_rate = discount_rates.get(disacount_tier.lower(), 0.0)
    final_price = price * (1 - discount_rate)
    return final_price


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. You have access to the product catalog tool and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price.\n"
                "You MUST call the get_product_price tool to get the real price of a product.\n"
                "2. NEVER guess or assume any discount.\n"
                "You MUST call the apply_discount tool AFTER you have received a price from get_product_price. Pass the exact price."
                "returned by get_product_price - do NOT pass a made-up number."
                "3. NEVER calcuate disacounts yourself using math."
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier,"
                "ask them which tier to use - do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration}:")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\n Final answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"    >> Tool selected: {tool_name} with args: {tool_args}")

        tool_to_call_fn = tools_dict.get(tool_name)

        if tool_to_call_fn is None:
            print(f"Error: Tool '{tool_name}' not found.")
            raise ValueError(f"Tool '{tool_name}' not found.")

        observation = tool_to_call_fn.invoke(tool_args)

        print(f"    >> Observation from tool '{tool_name}': {observation}")
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("Running agent loop with LangChain tool calling...")
    question = "What is the price of a laptop with a gold discount?"
    run_agent(question)
