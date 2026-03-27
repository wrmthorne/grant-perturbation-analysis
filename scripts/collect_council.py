import asyncio
import re
from collections import defaultdict

from inspect_ai import task
from inspect_ai.solver import TaskState, solver
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.model._generate_config import active_generate_config, set_active_generate_config
from pydantic import ValidationError

from collect_utils import (
    BASE_SYSTEM_PROMPT, REVIEW_CRITERIA, SCORE_DESCRIPTIONS,
    get_review_response_schema, build_config, create_task,
)
from inspect_utils import Review, convert_to_grouped_sections



PERSONAS = {
    "cost_analyst": """\
You are financially minded: someone who pays particular attention to value for money, resource allocation, and cost-effectiveness. While you must still provide a comprehensive review covering all aspects, you are especially critical of:

- Budget justification and whether costs are reasonable and necessary
- Efficient use of resources and personnel
- Whether the proposed outcomes justify the investment
- Risk of cost overruns or inefficient spending
- Whether similar outcomes could be achieved with fewer resources""",

    "ethics_assessor": """\
You are ethically minded: someone who emphasizes responsible research practices and societal implications. While you must still provide a comprehensive review covering all aspects, you are especially attentive to:

- Research ethics and responsible innovation
- Data privacy, security, and governance considerations
- Potential societal impacts, both positive and negative
- Inclusivity and equitable access to research benefits
- Environmental sustainability and long-term consequences""",

    "tech_evangelist": """\
You are a tech evangelist: you value innovation, cutting-edge approaches, and technological advancement. While you must still provide a comprehensive review covering all aspects, you are especially excited by:

- Novel technologies and innovative methodologies
- Potential for breakthrough discoveries or transformative applications
- Technical sophistication and ambition
- Integration of emerging technologies
- Opportunities to push boundaries and challenge conventions""",

    "methodological_sceptic": """\
You are a methodological skeptic who scrutinizes research design and scientific rigor. While you must still provide a comprehensive review covering all aspects, you are especially critical of:

- Methodological soundness and appropriateness
- Validity of proposed approaches and assumptions
- Adequacy of controls, validation strategies, and error analysis
- Whether claims are supported by the proposed methods
- Potential confounds, biases, or limitations in the research design""",

    "impact_champion": """\
You are an impact champion who focuses on real-world applications and broader benefits. While you must still provide a comprehensive review covering all aspects, you are especially interested in:

- Pathways to impact and how outcomes will be translated
- Engagement with stakeholders, industry, or end-users
- Potential for economic, social, or cultural benefits
- Plans for dissemination and knowledge exchange
- Long-term sustainability and scalability of impacts""",

    "chair": """\
You are a synthesizer who excels at integrating diverse expert opinions. You are particularly attuned to:

- When disagreement reflects genuine trade-offs versus differences in evidence quality
- The credibility and rigor behind different viewpoints, not just their conviction
- Patterns that emerge across independent assessments
- When a minority position raises valid concerns that consensus overlooks
- Proportional weighting - giving appropriate influence to well-reasoned arguments
- Distinguishing between complementary perspectives and genuine contradictions"""
}


def _parse_ranking_from_text(ranking_text: str) -> list[str]:
    if "FINAL RANKING:" in ranking_text:
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            numbered_matches = re.findall(r'\d+\.\s*Review [A-E]', ranking_section)
            if numbered_matches:
                return [re.search(r'Review [A-E]', m).group() for m in numbered_matches]
            return re.findall(r'Review [A-E]', ranking_section)
    return re.findall(r'Review [A-E]', ranking_text)


@solver
def council_solver():
    async def solve(state, generate):
        proposal_content = state.input_text
        files = state.metadata.get("files", {})
        grouped_sections = convert_to_grouped_sections(files, return_summary=True)
        summary = grouped_sections.pop("summary", "")

        base_config = active_generate_config()
        review_config = GenerateConfig(
            **base_config.model_dump(exclude={"response_schema"}),
            response_schema=get_review_response_schema(),
        )

        # STAGE 1: Individual reviews from all personas
        async def review_with_persona(persona_name, persona_prompt):
            system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{persona_prompt}"
            user_prompt = f"""\
For the provided grant proposal, give a score between 1 and 6 accompanied by a detailed justification for your score.

{SCORE_DESCRIPTIONS}

{REVIEW_CRITERIA}

{proposal_content}

Provide your final assessment as a JSON object with "score" (integer 1-6) and "explanation" (string) fields."""

            persona_state = TaskState(
                model=state.model,
                sample_id=state.sample_id,
                epoch=state.epoch,
                input=state.input,
                messages=[
                    ChatMessageSystem(content=system_prompt),
                    ChatMessageUser(content=user_prompt)
                ],
                metadata=state.metadata
            )
            persona_state = await generate(persona_state)

            try:
                review = Review.model_validate_json(persona_state.output.completion)
                return {"persona": persona_name, "score": review.score, "explanation": review.explanation}
            except ValidationError:
                return None

        set_active_generate_config(review_config)
        stage1_results = await asyncio.gather(*[
            review_with_persona(name, prompt) for name, prompt in PERSONAS.items()
        ])
        stage1_reviews = [r for r in stage1_results if r is not None]

        if not stage1_reviews:
            return state

        # STAGE 2: Peer ranking
        labels = [chr(65 + i) for i in range(len(stage1_reviews))]
        label_to_persona = {f"Review {label}": review['persona'] for label, review in zip(labels, stage1_reviews)}

        reviews_text = "\n\n".join([
            f"Review {label} (Score: {review['score']}):\n{review['explanation']}"
            for label, review in zip(labels, stage1_reviews)
        ])

        ranking_prompt = f"""\
You are evaluating different reviews of the same grant proposal.

## Summary

{summary}

## Reviews

{reviews_text}

Your task:
1. First, evaluate each review individually. For each review, explain what it does well and what weaknesses it has in its assessment.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the reviews from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the review label (e.g., "1. Review A")
- Do not add any other text or explanations in the ranking section

Example format:
Review A provides comprehensive coverage but...
Review B is overly critical on...

FINAL RANKING:
1. Review C
2. Review A
3. Review B

Now provide your evaluation and ranking:"""

        async def rank_with_persona(persona_name, persona_prompt):
            system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{persona_prompt}"
            persona_state = TaskState(
                model=state.model,
                sample_id=state.sample_id,
                epoch=state.epoch,
                input=state.input,
                messages=[
                    ChatMessageSystem(content=system_prompt),
                    ChatMessageUser(content=ranking_prompt)
                ],
                metadata=state.metadata
            )
            persona_state = await generate(persona_state)
            return {
                "persona": persona_name,
                "ranking_text": persona_state.output.completion,
                "parsed_ranking": _parse_ranking_from_text(persona_state.output.completion)
            }

        set_active_generate_config(base_config)
        stage2_results = await asyncio.gather(*[
            rank_with_persona(name, prompt) for name, prompt in PERSONAS.items()
        ])

        # Calculate aggregate rankings
        persona_positions = defaultdict(list)
        for ranking in stage2_results:
            for position, label in enumerate(ranking['parsed_ranking'], start=1):
                if label in label_to_persona:
                    persona_positions[label_to_persona[label]].append(position)

        aggregate_rankings = sorted([
            {"persona": persona, "average_rank": round(sum(pos) / len(pos), 2), "rankings_count": len(pos)}
            for persona, pos in persona_positions.items() if pos
        ], key=lambda x: x['average_rank'])

        # STAGE 3: Chairman synthesis
        set_active_generate_config(review_config)

        stage1_text = "\n\n".join([
            f"**{review['persona'].replace('_', ' ').title()} Reviewer** (Score: {review['score']})\n{review['explanation']}"
            for review in stage1_reviews
        ])
        stage2_text = "\n\n".join([
            f"**{ranking['persona'].replace('_', ' ').title()} Ranking:**\n{ranking['ranking_text']}"
            for ranking in stage2_results
        ])
        aggregate_text = "\n".join([
            f"{i + 1}. {agg['persona'].replace('_', ' ').title()} (Average rank: {agg['average_rank']})"
            for i, agg in enumerate(aggregate_rankings)
        ])

        chairman_prompt = f"""\
Multiple expert reviewers have provided reviews and then ranked each other's assessments. Your task as Chairman is to synthesize all of this information into a single score (1-6) and explanation. Consider:

- The individual reviews and their insights
- The peer rankings and what they reveal about review quality
- Any patterns of agreement or disagreement
- The aggregate rankings showing which perspectives were most valued

## Stage 1 - Individual Reviews

{stage1_text}

## Stage 2 - Peer Rankings

{stage2_text}

## Aggregate Rankings (Best to Worst)

{aggregate_text}

Provide your final assessment as a JSON object with "score" (integer 1-6) and "explanation" (string) fields."""

        state.messages = [
            ChatMessageSystem(content=BASE_SYSTEM_PROMPT),
            ChatMessageUser(content=chairman_prompt)
        ]
        state = await generate(state)

        state.metadata["stage1_reviews"] = stage1_reviews
        state.metadata["stage2_rankings"] = stage2_results
        state.metadata["aggregate_rankings"] = aggregate_rankings
        state.metadata["label_to_persona"] = label_to_persona

        return state

    return solve


@task
def council_reviews(
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
    )
    return create_task(
        council_solver, config,
        include_original=include_original,
        include_perturbed=include_perturbed,
        target_proposals=target_proposals,
        epochs=epochs,
        limit=limit,
        shuffle=shuffle,
    )