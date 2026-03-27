from inspect_ai import Task
from inspect_ai.scorer import Score, scorer, mean, std
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema
from pydantic import ValidationError

from inspect_utils import Review, create_dataset



BASE_SYSTEM_PROMPT = """\
You are an expert reviewer for the Engineering and Physical Sciences Research Council (EPSRC). Your scores, comments, and actions will be used to determine the whether a given EPSRC research grant proposal is funded or unfunded. Your comments and decisions must depend only on the evidence provided, citing examples in the provided documents where appropriate. Take a firm but fair approach."""

REVIEW_CRITERIA = """\
## Review Criteria

We'll only be able to use your review if it meets the following criteria:

- you've included enough information to help UKRI staff and panellists make an informed judgement on the application
- your comments are only based on information that's included in the application
- you have not reviewed the application negatively because of any equality, diversity and inclusion requirements (for example, decisions to work part-time or past absences for health reasons)
- your comments are not speculative, inflammatory or damaging to applicants
- you have not used journal metrics, conference rankings or personal metrics as a substitute measure for assessing the applicants' contributions
- you do not have a conflict of interest with the application and have not revealed your identity"""

SCORE_DESCRIPTIONS = """\
## Score Descriptions

6 - Exceptional: The application is outstanding. It addresses all of the assessment criteria and meets them to an exceptional level.
5 - Excellent: The application is very high quality. It addresses most of the assessment criteria and meets them to an excellent level. There are very minor weaknesses.
4 - Very good: The application demonstrates considerable quality. It meets most of the assessment criteria to a high level. There are minor weaknesses.
3 - Good: The application is of good quality. It meets most of the assessment criteria to an acceptable level, but not across all aspects of the proposed activities. There are weaknesses.
2 - Weak: The application is not sufficiently competitive. It meets some of the assessment criteria to an adequate level. There are, however, significant weaknesses.
1 - Poor: The application is flawed or unsuitable quality for funding. It does not meet the assessment criteria to an adequate level."""


def get_review_response_schema():
    return ResponseSchema(name="review", json_schema=json_schema(Review), strict=True)


@scorer(metrics=[mean(), std()])
def review_score():
    async def score(state, target):
        try:
            review = Review.model_validate_json(state.output.completion)
            metadata = {
                "predicted_score": review.score,
                "predicted_explanation": review.explanation,
            }
            return Score(
                value=float(review.score),
                answer=state.output.completion,
                metadata=state.metadata | metadata,
            )
        except ValidationError as e:
            return Score(
                value=float('nan'),
                answer=state.output.completion,
                explanation=f"Validation error: {e}",
                metadata=state.metadata,
            )
    return score


def create_task(solver_fn, config, include_original=True, include_perturbed=True,
                epochs=5, limit=None, shuffle=True, target_proposals=""):
    return Task(
        dataset=create_dataset(include_original=include_original, include_perturbed=include_perturbed, target_proposals=target_proposals),
        config=config,
        solver=[solver_fn()],
        scorer=review_score(),
        epochs=epochs,
        limit=limit,
        shuffle=shuffle,
    )


def build_config(temperature=None, top_k=None, top_p=None, reasoning_effort=None,
                 response_schema=None):
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_k is not None:
        kwargs["top_k"] = top_k
    if top_p is not None:
        kwargs["top_p"] = top_p
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if response_schema is not None:
        kwargs["response_schema"] = response_schema
    return GenerateConfig(**kwargs)