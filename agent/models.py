from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()  # Load environment variables from .env file

model = init_chat_model(model = "deepseek-chat")  # Initialize the chat model using the API key from .env
