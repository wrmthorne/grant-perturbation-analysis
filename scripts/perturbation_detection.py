from inspect_ai.solver import solver, generate
from inspect_ai.scorer import Score, scorer, accuracy, mean
from inspect_ai.util import json_schema
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_utils import load_all_proposals, PerturbationDetection
from inspect_ai.analysis import samples_df, SampleSummary, SampleScores, EvalTask
from inspect_ai.model import GenerateConfig, ChatMessageUser, ChatMessageSystem, ResponseSchema
import pandas as pd
from functools import partial
import re


JUDGE_SYSTEM = """\
You are an expert evaluator assessing whether an LLM-generated grant proposal review correctly identifies a known perturbation (intentional flaw) that was introduced into the proposal.

You will be given:
1. A description of the perturbation that was applied
2. The exact diff showing what changed between the original and perturbed proposal
3. The LLM-generated review of the perturbed proposal

Your task is to determine whether the review identifies the perturbation or a direct consequence of it."""

JUDGE_PROMPT = """\
You are evaluating whether a reviewer identified an introduced error in a funding proposal.

## Context

A genuine EPSRC proposal was adversarially modified with the following perturbation:
{perturbation_description}

## File Changes

{diff}

## Review Text
{review_text}

## Task
Evaluate whether the review identifies the perturbation. Consider:
- Does the review explicitly mention the specific issue introduced?
- Does the review identify a direct, obvious consequence of the perturbation?
- Vague or generic criticisms that could apply to any proposal do NOT count

Award exactly one label:
- C (Correct): Review explicitly identifies and discusses the introduced error or direct consequences
- P (Partial): Review makes vague or incomplete reference to issues from the perturbation
- I (Incorrect): Review fails to acknowledge the error or only mentions it tangentially

Respond with a JSON with two string fields: explanation and verdict."""


@solver
def annotate_solver():
    async def solve(state, generate):
        state = await generate(state)
        return state

    return solve


@scorer(metrics=[accuracy(), mean()])
def perturbation_scorer():
    async def score(state, target):
        response = PerturbationDetection.model_validate_json(state.output.completion)

        return Score(
            value=response.verdict.strip(),
            answer=state.output.completion,
            explanation=response.explanation.strip(),
            metadata=state.metadata
        )

    return score


def extract_fields(content, fields: list[str]) -> dict[str, str | None]:
    """Extract named fields from potentially malformed JSON"""
    result = {field: None for field in fields}

    if pd.isna(content):
        return result

    content = content.replace('“”', '"') # Replace non-standard double-quotes
    content = content.replace("‘’", "'") # Replace non-standard single-quotes

    for field in fields:
        match = re.search(rf'"{field}"\s*:\s*"(.*?)(?:"\s*[,}}]|"\s*$)', content, re.DOTALL)
        if match:
            value = match.group(1).replace('\\"', '"')
            result[field] = value
        else:
            # Try unquoted value (for booleans, numbers, etc.)
            match = re.search(rf'"{field}"\s*:\s*([^,}}\s]+)', content)
            if match:
                result[field] = match.group(1)

    return result


def build_dataset(log_dir: str = "logs_20b"):
    task_name_mapping = {
        "collect_reviews": "baseline",
        "collect_sectioned": "sectioned",
        "collect_council": "council"
    }

    extract_score_explanation = partial(extract_fields, fields=["score", "explanation"])
    samples = samples_df(log_dir, parallel=True, columns=SampleSummary + SampleScores + EvalTask)
    samples[["score", "explanation"]] = samples["score_review_score_answer"].apply(extract_score_explanation).apply(
        pd.Series)

    review_systems = samples["task_name"].map(task_name_mapping)
    samples.insert(3, "review_system", review_systems)
    samples.set_index("sample_id", inplace=True)
    perts_mask = samples["metadata_sample_type"] != "original"

    proposals = load_all_proposals()
    proposal_map = {p.id: p for p in proposals}

    sample_list = []
    for sample_idx in samples[perts_mask].index:
        row = samples.loc[sample_idx]
        original_task_args = {
            col_name: col_value for col_name, col_value in row.items() if str(col_name).startswith("task_arg")
        }
        proposal = proposal_map.get(row['metadata_proposal_id'])
        if not proposal:
            continue

        perturbation = next(
            (p for p in proposal.perturbations if p.name == row['metadata_perturbation_name']),
            None
        )
        if not perturbation:
            continue

        diffs_text = "\n\n".join(
            f"### {filename}\n{diff}"
            for filename, diff in perturbation.diffs.items()
        )

        prompt = JUDGE_PROMPT.format(
            perturbation_description=perturbation.description,
            diff=diffs_text,
            review_text=row["explanation"],
        )
        messages_list = [
            ChatMessageSystem(content=JUDGE_SYSTEM),
            ChatMessageUser(content=prompt)
        ]

        sample_list.append(Sample(
            id=sample_idx,
            input=messages_list,
            target="",
            metadata={
                "perturbation_description": perturbation.description,
                "diff": diffs_text,
                "review_text": row["explanation"],
                "verdict": None,
                "original_task_args": original_task_args,
            }
        ))

    return sample_list


@task
def annotate_perturbations(
    epochs: int = 1,
    limit: int | None = None,
    shuffle: bool = True,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    reasoning_effort: str | None = None,
):
    config_kwargs = {}
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    if top_k is not None:
        config_kwargs["top_k"] = top_k
    if top_p is not None:
        config_kwargs["top_p"] = top_p
    if reasoning_effort is not None:
        config_kwargs["reasoning_effort"] = reasoning_effort

    config = GenerateConfig(
        **config_kwargs,
        response_schema=ResponseSchema(
            name="review",
            json_schema=json_schema(PerturbationDetection),
            strict=True,
        )
    )

    return Task(
        config=config,
        dataset=build_dataset(),
        solver=[generate()],
        scorer=[
            perturbation_scorer()
        ],
        epochs = epochs,
        limit = limit,
        shuffle = shuffle,
    )