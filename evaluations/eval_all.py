import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    GEval,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall, ToolCallParams

if __package__:
    from .dataset import dataset
    from .model import eval_model
else:
    from dataset import dataset
    from model import eval_model


SUCCESSFUL_TASK_DESCRIPTION = """
If the username is missing, ask the user to provide it.
For a refund request, identify the customer and the requested purchases, and
check that each purchase is no more than 30 days old and is not in the future.
Refund only the items requested by the user. If the customer or requested item
cannot be found, do not issue a refund ticket and explain why.
Once the necessary information is available, tell the user whether a refund
ticket was issued and its amount, or explain why no refund was issued.
For an information-only request, answer the question without issuing a ticket.
"""


def build_metrics():
    """Keep the thresholds and evaluator model used by the individual scripts."""
    # These three need the LangGraph trace and are evaluated by evals_iterator.
    trace_metrics = [
        StepEfficiencyMetric(
            threshold=0.8, model=eval_model, include_reason=True,
        ),
        PlanQualityMetric(
            threshold=0.8, model=eval_model, include_reason=True,
        ),
        PlanAdherenceMetric(
            threshold=0.8, model=eval_model, include_reason=True,
        ),
    ]
    # TaskCompletionMetric prefers a trace but falls back to input/actual_output,
    # so it is judged on the clean structured answer instead of the raw dump.
    single_turn_metrics = [
        TaskCompletionMetric(
            threshold=0.8,
            model=eval_model,
            task=SUCCESSFUL_TASK_DESCRIPTION,
            include_reason=True,
        ),
        ArgumentCorrectnessMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
        ),
        ToolCorrectnessMetric(
            threshold=0.9, 
            model=eval_model,
            include_reason=True,
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS]
        ),
        GEval(
            name="Answer Correctness",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            evaluation_steps=[
                "Check that the actual output and expected output have the same "
                "logical conclusion about the user's request.",
            ],
            threshold=0.8,
            model=eval_model,
            verbose_mode=True,
        ),
    ]
    return trace_metrics, single_turn_metrics


def main():
    load_dotenv()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
    from agent import agent
    from tools import (
        calculate_right_refund,
        get_today_date,
        get_user_purchase_history,
        issue_refund_ticket,
    )

    # response_format=Answer is implemented as a pseudo tool call named "Answer";
    # only real agent tools count towards the tool metrics.
    agent_tool_names = {
        get_user_purchase_history.name,
        calculate_right_refund.name,
        issue_refund_ticket.name,
        get_today_date.name,
    }

    trace_metrics, single_turn_metrics = build_metrics()
    cache_config = CacheConfig(use_cache=False, write_cache=False)
    display_config = DisplayConfig(inspect_after_run=False)
    # DeepSeek allows 5 concurrent requests on this account; deepeval defaults
    # to 20, which can trigger HTTP 429.
    async_config = AsyncConfig(max_concurrent=4)
    test_cases = []

    # The callback captures the trace required by the three agentic metrics.
    for golden in dataset.evals_iterator(
        metrics=trace_metrics,
        cache_config=cache_config,
        display_config=display_config,
        async_config=async_config,
    ):
        response = agent.invoke(
            {"messages": [{"role": "user", "content": golden.input}]},
            config={
                "configurable": {"thread_id": str(uuid4())},
                "callbacks": [CallbackHandler()],
            },
        )

        tools_called = []
        for message in response["messages"]:
            for tool_call in getattr(message, "tool_calls", []) or []:
                if tool_call["name"] not in agent_tool_names:
                    continue
                tools_called.append(
                    ToolCall(
                        name=tool_call["name"],
                        input_parameters=tool_call["args"],
                    )
                )

        # Reuse the same response for the four single-turn metrics. Keep these
        # cases local so repeated runs do not accumulate dataset.test_cases.
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=response["structured_response"].answer,
                expected_output=golden.expected_output,
                tools_called=tools_called,
                expected_tools=golden.expected_tools,
            )
        )

    evaluate(
        test_cases=test_cases,
        metrics=single_turn_metrics,
        cache_config=cache_config,
        display_config=display_config,
        async_config=async_config,
    )
    print(f"\nCompleted all 7 metrics for {len(test_cases)} test cases.")


if __name__ == "__main__":
    main()