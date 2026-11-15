import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(
    0, 
    str(Path(__file__).resolve().parents[1] / "agent")
)

from agent import agent
from dataset import dataset
from model import eval_model


from deepeval.metrics import GEval
from deepeval.integrations.langchain import CallbackHandler
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval import evaluate
from dotenv import load_dotenv


load_dotenv()


for golden in dataset.goldens:
    response = agent.invoke({"messages": [{"role": "user", "content": golden.input}]}, 
                            config={"configurable": {"thread_id": str(uuid4())},
                            "callbacks":[CallbackHandler()]})

    actual_output = response["structured_response"].answer

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
    )

    dataset.add_test_case(test_case)


mertrics = [
    GEval(
        name="Answer correctness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps = [
            "make sure that the actual output is correct and matches the expected output",
        ],
        threshold=0.8,
        model = eval_model,
        verbose_mode=True,
    )
]


evaluate(
    test_cases=dataset.test_cases,
    metrics=mertrics,
)






