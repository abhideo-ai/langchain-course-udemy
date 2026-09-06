"""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

import re
import inspect
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()


import ollama
from langsmith import traceable

MAX_ITERATIONS = 10

MODEL = "qwen3:1.7b"


@traceable(run_type="tool")
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


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold, platinum
    """
    print(
        f"    >> Executing apply_discount(price={price}, disacount_tier={discount_tier})"
    )
    price = float(price)
    discount_rates: Dict[str, float] = {
        "bronze": 0.05,
        "silver": 0.10,
        "gold": 0.15,
        "platinum": 0.20,
    }
    discount_rate = discount_rates.get(discount_tier.lower(), 0.0)
    final_price = price * (1 - discount_rate)
    return final_price



tools_dict = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

def get_tool_descriptions(tools_dict: Dict[str, str]) -> List[str]:
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        descriptions.append(f"{tool_name}{signature} = {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools_dict)
tool_names = ", ".join(tools_dict.keys())

react_prompt = f"""
STRICT RULES - you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call the get_product_price tool to get the real price of a product.
2. NEVER guess or assume any discount. You MUST call the apply_discount tool AFTER you have received a price from get_product_price. 
Pass the exact price returned by get_product_price - do NOT pass a made-up number.
3. NEVER calculate disacounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one.
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""
pass


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(
        model=model,
        messages=messages,
        options=options,
    )


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    prompt = react_prompt.format(question=question)
    scratch_pad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration}:")

        full_prompt = prompt + scratch_pad

        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content

        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"\t[Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(f"\t[Parsing] ERROR: Could not parse Action/Action Input from LLM Output")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"    >> Tool selected: '{tool_name}' with args: {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args ]

        print(f"\t[Tool Executing] {tool_name}({args})...")

        if tool_name not in tools_dict:
            observation = f"Error: Tool '{tool_name}' not recognized. Available tools: {list[str](tools_dict.keys())}"
        else:
            observation = str(tools_dict[tool_name](*args))

        print(f"    >> Observation from tool '{tool_name}': {observation}")
        scratch_pad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("Running agent loop with LangChain tool calling...")
    question = "What is the price of a laptop with a gold discount?"
    run_agent(question)
