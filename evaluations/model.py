from pathlib import Path

from dotenv import load_dotenv
from deepeval.models.llms.deepseek_model import DeepSeekModel

# DeepSeekModel reads DEEPSEEK_API_KEY when it is constructed, so .env must be
# loaded before the model is created. Resolve it relative to the project root so
# the scripts work from any working directory.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Judge model for the evaluation metrics. The API key is read from .env
# (DEEPSEEK_API_KEY) by deepeval's settings.
#
# "deepseek-chat", "deepseek-reasoner" and "deepseek-v4-flash" all resolve to
# the same upstream model ("deepseek-flash"). Use "deepseek-v4-pro" for a
# stronger (and slower/more expensive) judge.
eval_model = DeepSeekModel(model="deepseek-v4-flash")
