from inspect_ai import task
from inspect_ai.solver import solver
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from collect_utils import (
    BASE_SYSTEM_PROMPT, REVIEW_CRITERIA, SCORE_DESCRIPTIONS,
    get_review_response_schema, build_config, create_task,
)


@solver
def baseline_solver():
    async def solve(state, generate):
        prompt = f"""\
For the provided grant proposal, give a score between 1 and 6 accompanied by a detailed justification for your score.

{SCORE_DESCRIPTIONS}

{REVIEW_CRITERIA}

{state.input_text}

Provide your final assessment as a JSON object with "score" (integer 1-6) and "explanation" (string) fields."""

        state.messages = [
            ChatMessageSystem(content=BASE_SYSTEM_PROMPT),
            ChatMessageUser(content=prompt)
        ]
        return await generate(state)

    return solve


@task
def collect_reviews(
        epochs: int = 5,
        limit: int | None = None,
        shuffle: bool = True,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        reasoning_effort: str | None = None,
        include_original: bool = True,
        include_perturbed: bool = True,
        target_proposals: str = "", # comma separated proposal names
):
    config = build_config(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        reasoning_effort=reasoning_effort,
        response_schema=get_review_response_schema(),
    )
    return create_task(
        baseline_solver, config,
        include_original=include_original,
        include_perturbed=include_perturbed,
        target_proposals=target_proposals,
        epochs=epochs,
        limit=limit,
        shuffle=shuffle,
    )