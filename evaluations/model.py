from deepeval.models.llms.deepseek_model import DeepSeekModel

# deepeval resolves a bare string like "deepseek-chat" to an OpenAI model, so
# pass a DeepSeek-backed LLM explicitly. The API key is read from .env
# (DEEPSEEK_API_KEY) by deepeval's settings.
eval_model = DeepSeekModel(model="deepseek-chat")
