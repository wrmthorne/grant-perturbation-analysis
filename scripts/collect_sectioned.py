import asyncio

from inspect_ai import task
from inspect_ai.solver import TaskState, solver
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.model._generate_config import active_generate_config, set_active_generate_config

from collect_utils import (
    BASE_SYSTEM_PROMPT, REVIEW_CRITERIA, SCORE_DESCRIPTIONS,
    get_review_response_schema, build_config, create_task,
)
from inspect_utils import convert_to_grouped_sections



SECTION_CRITERIA = {
    "vision_approach": """\
The project's vision...
- is of excellent quality and importance within or beyond the field(s) or area(s)
- has the potential to advance current understanding, or generate new knowledge, thinking or discovery within or beyond the field or area
- is timely given current trends, context, and needs
- impacts world-leading research, society, the economy, or the environment

The proposed approach...
- is effective and appropriate to achieve their objectives
- is feasible, and comprehensively identifies any risks to delivery and how they will be managed
- uses a clear and transparent methodology (if applicable)
- summarises the previous work and describes how this will be built upon and progressed (if applicable)
- will maximise translation of outputs into outcomes and impacts
- describes how the PI, and if applicable the PI's team, research environment (in terms of the place, its location, and relevance to the project) will contribute to the success of the work""",

    "team_capability": """\
The team has...
- the relevant experience (appropriate to career stage) to deliver the proposed work
- the right balance of skills and expertise to cover the proposed work
- the appropriate leadership and management skills to deliver the work and your approach to develop others
- contributed to developing a positive research environment and wider community""",

    "ethics": "Is the proposal ethical?",

    "funding_resources": """\
- all resources are appropriate
- the project will make optimal use of resources to achieve its outcomes"""
}


@solver
def sectioned_solver():
    async def solve(state, generate):
        files = state.metadata.get("files", {})
        grouped_sections = convert_to_grouped_sections(files, return_opportunity=True)
        opportunity = grouped_sections.pop("opportunity", "")

        opportunity_info = ""
        for section in opportunity.sections:
            if "looking for" in section.title.lower():
                opportunity_info = f"""\
## Opportunity

### {section.title}

{section.content}"""
                break

        base_config = active_generate_config()
        review_config = GenerateConfig(
            **base_config.model_dump(exclude={"response_schema"}),
            response_schema=get_review_response_schema(),
        )

        async def review_section(section_name: str, content: str) -> dict[str, str] | None:
            prompt = f"""\
You are given a single section from a proposal and the opportunity it was submitted to. Review this section by detailed, critical comments on its quality, strengths, and weaknesses. Do not provide a score.

{REVIEW_CRITERIA}

You should think carefully about whether the proposal meets the following:

{SECTION_CRITERIA[section_name]}
{opportunity_info}
## Proposal Section

{content}

Provide a thorough assessment of this section."""

            section_state = TaskState(
                model=state.model,
                sample_id=state.sample_id,
                epoch=state.epoch,
                input=state.input,
                messages=[
                    ChatMessageSystem(content=BASE_SYSTEM_PROMPT),
                    ChatMessageUser(content=prompt)
                ],
                metadata=state.metadata
            )

            try:
                section_state = await generate(section_state)
                return {"section": section_name, "comments": section_state.output.completion}
            except Exception as e:
                print(f"Error while generating section: {e}")

        tasks = [review_section(name, content) for name, content in grouped_sections.items() if content]
        results = await asyncio.gather(*tasks)
        section_reviews = [r for r in results if r is not None]

        section_reviews_text = "\n\n".join([
            f"**{review['section'].replace('_', ' ').title()}**\n{review['comments']}"
            for review in section_reviews
        ])

        synthesis_prompt = f"""\
You are synthesizing multiple section-specific reviews into a final comprehensive review of a grant proposal.

{SCORE_DESCRIPTIONS}

{REVIEW_CRITERIA}

## Section Specific Reviews
{section_reviews_text}

Your task is to synthesize all of this information into a single, comprehensive review with a score between 1 and 6. Consider:
- The insights from each section review
- Any patterns of strengths or weaknesses across sections
- The overall coherence and quality of the proposal
- The review criteria:

Provide a clear, well-reasoned final review as a JSON object with "score" (integer 1-6) and "explanation" (string) fields."""

        state.messages = [
            ChatMessageSystem(content=BASE_SYSTEM_PROMPT),
            ChatMessageUser(content=synthesis_prompt)
        ]

        set_active_generate_config(review_config)
        state = await generate(state)
        state.metadata["section_reviews"] = section_reviews

        return state

    return solve


@task
def sectioned_reviews(
        epochs: int = 5,
        limit: int | None = None,
        shuffle: bool = True,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        reasoning_effort: str | None = None,
        include_original: bool = True,
        include_perturbed: bool = True,
        target_proposals: str = "",  # comma separated proposal names
):
    config = build_config(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        reasoning_effort=reasoning_effort,
    )
    return create_task(
        sectioned_solver, config,
        include_original=include_original,
        include_perturbed=include_perturbed,
        target_proposals=target_proposals,
        epochs=epochs,
        limit=limit,
        shuffle=shuffle,
    )