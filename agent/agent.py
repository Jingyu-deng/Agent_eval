from models import model
from tools import get_user_purchase_history, calculate_right_refund, issue_refund_ticket, get_today_date

from dotenv import load_dotenv

from langchain.agents import create_agent
from prompts import agent_system_prompts
from objects import Answer, PurchaseItem

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain.messages import HumanMessage
from uuid import uuid4

load_dotenv()  # Load environment variables from .env file

agent = create_agent(
    model=model,
    tools=[get_user_purchase_history, calculate_right_refund, issue_refund_ticket, get_today_date],
    system_prompt=agent_system_prompts,
    response_format=Answer,
    checkpointer=InMemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=[Answer])
    )
)


def run_conversation():
    config = {
        "configurable": {
            "thread_id": str(uuid4()),  # Generate a unique conversation ID
        }
    }


    question = HumanMessage(content="I want a refund for my purchase of the item 'Wireless Mouse'. My username is 'john_doe'.")
    conversation = True

    while conversation:
        response = agent.invoke({"messages": [question]}, config=config)

        structured_response = response["structured_response"]

        print(f"Agent Response: {structured_response}")

        conversation = structured_response.conversation

        if conversation:
            user_input = input("reply: \n")
            question = HumanMessage(content=user_input)

    print("Conversation ended.")


if __name__ == "__main__":
    run_conversation()










