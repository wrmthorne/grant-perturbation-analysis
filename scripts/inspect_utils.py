from pathlib import Path
from typing import Any
import hashlib
import json
from difflib import unified_diff
import warnings
from collections import defaultdict

from inspect_ai.dataset import Sample

from data_models import *


ROOT_PATH = Path(__file__).parent

# Application overview questions used in each section group
grouped_sections = {
    "vision_approach": ["references"], # + entire vision and approach document
    "team_capability": ["capability", "core team", "project partners", "facilities", "references"],
    "funding_resources": ["core team", "resources and cost", "facilities", "references"],
    "ethics": ["ethics"] # ... and any answered questions after ethics in application-overview
}

sections_to_groups = {
    "vision": "vision_approach",
    "approach": "vision_approach",
    "capability": "team_capability",
    "cost": "funding_resources",
    "ethics": "ethics"
}


def parse_header_and_questions(content):
    sections = content.split('\n\n---\n\n')
    if not sections:
        return None, []

    first_section = sections[0].strip()
    lines = first_section.split('\n')
    title = lines[1 if lines[0].startswith('# ') else 0].strip()

    questions = []
    for section in sections[1:]:
        if not (section := section.strip()):
            continue

        lines = section.split('\n')
        first_line = lines[0].strip()

        question_num, question_title = None, ""
        if first_line.startswith('## '):
            header = first_line[3:].strip()
            if '. ' in header:
                num_part, title_part = header.split('. ', 1)
                try:
                    question_num = int(num_part)
                    question_title = title_part
                except ValueError:
                    question_title = header
            else:
                question_title = header

        question_lines = []
        response_lines = []
        in_question = False

        for line in lines[1:]:
            if line.strip().startswith('>'):
                in_question = True
                question_lines.append(line.strip()[1:].strip())
            elif in_question and not line.strip().startswith('>'):
                in_question = False
                response_lines.append(line)
            elif not in_question:
                response_lines.append(line)

        question_text = '\n'.join(question_lines).strip()
        response_text = '\n'.join(response_lines).strip()

        # Skip if empty string or NA
        if not response_text or response_text.upper() in ("NA", "N/A"):
            continue

        questions.append(Question(question_num=question_num, question_title=question_title,
                                  question_text=question_text, response_text=response_text))

    return title, questions


def parse_application_overview(content):
    sections = content.split('\n\n---\n\n')
    if not sections:
        return None

    first_section = sections[0].strip()
    lines = first_section.split('\n')
    title = lines[1 if lines[0].startswith('# ') else 0].strip()

    summary = ""
    for section in sections[1:]:
        if not (section := section.strip()):
            continue

        if '### Summary' in section:
            parts = section.split('### Summary', 1)
            if len(parts) > 1:
                summary_part = parts[1].strip()
                summary = summary_part.split('\n####')[0].strip()
                break

    _, questions = parse_header_and_questions(content)
    return ApplicationOverview(title=title, summary=summary, questions=questions)


def parse_fit_to_opportunity(content):
    title, questions = parse_header_and_questions(content)
    return FitToOpportunity(title=title, sections=questions) if title is not None else None


def parse_vision_and_approach(content):
    sections = content.split('\n\n---\n\n')
    if len(sections) != 2:
        return None

    lines = sections[0].strip().split('\n', 1)
    title, vision = "", sections[0].strip()
    if lines[0].startswith('# '):
        title = lines[0][2:].strip()
        vision = lines[1].strip() if len(lines) > 1 else ""

    return VisionAndApproach(title=title, vision=vision, approach=sections[1].strip())


def parse_opportunity(content):
    sections = content.split('\n\n---\n\n')
    if not sections:
        return None

    lines = sections[0].strip().split('\n')
    title, metadata, content_start = "", {}, 0

    if lines[0].startswith('# '):
        title = lines[0][2:].strip()
        content_start = 1

    if content_start < len(lines) and lines[content_start].startswith('|'):
        i = content_start
        while i < len(lines) and lines[i].startswith('|'):
            i += 1

        for row in lines[content_start:i][2:]:
            parts = [p.strip() for p in row.split('|')[1:-1]]
            if len(parts) >= 2 and (key := parts[0].rstrip(':').strip()):
                metadata[key] = parts[1].strip()

    opportunity_sections = []
    for section in sections[1:]:
        if not (section := section.strip()):
            continue

        lines = section.split('\n', 1)
        section_title, section_content = "", section
        if lines[0].startswith('## '):
            section_title = lines[0][3:].strip()
            section_content = lines[1].strip() if len(lines) > 1 else ""

        opportunity_sections.append(OpportunitySection(title=section_title, content=section_content))

    return Opportunity(title=title, metadata=metadata, sections=opportunity_sections)


def parse_review(content):
    sections = content.split('\n\n---\n\n')
    if not sections:
        return None

    *split_sections, score = [section.strip().split('\n\n', maxsplit=1) for section in sections]
    score = int(score[-1].split("/")[0].strip())

    grouped_comments = defaultdict(str)
    for title, content in split_sections:
        for search_term, section_key in sections_to_groups.items():
            if search_term in title.lower():
                grouped_comments[section_key] += f"\n\n{content.strip()}"
                grouped_comments[section_key] = grouped_comments[section_key].strip()

    return SectionedReview(
        score=score,
        sections=[SectionComments(title=title, comments=comments)
                        for title, comments in grouped_comments.items()],
    )


def _get_questions_by_title(questions: list[Question], titles: list[str]) -> list[Question]:
    return [q for q in questions if any(title.lower() in q.question_title.lower() for title in titles)]


def _format_questions(questions: list[Question]) -> str:
    return "\n\n".join([f"## {q.question_title}\n\n{q.response_text}" for q in questions])


def _build_section_content(
        app_overview: ApplicationOverview | None,
        vision_approach: VisionAndApproach | None,
        section_type: str
) -> str:
    parts = []

    if section_type == "vision_approach":
        if vision_approach:
            parts.append(f"## Vision\n\n{vision_approach.vision}\n\n## Approach\n\n{vision_approach.approach}")

    if app_overview:
        if not section_type == "vision_approach":
            parts.append(f"## Summary\n\n{app_overview.summary}")

        questions = _get_questions_by_title(app_overview.questions, grouped_sections.get(section_type))
        if questions:
            parts.append(_format_questions(questions))

    return "\n\n".join(parts)


def load_all_proposals(target_proposals: list[str] | None = None) -> list[ProposalData]:
    """target_proposals e.g. ["APP_A", "APP_B"] or [] or ..."""
    if not target_proposals:
        target_proposals = []

    proposals = []

    for app_dir in sorted(ROOT_PATH.glob("APP*")):
        if not app_dir.is_dir() or (target_proposals and not any(p.lower() in str(app_dir).lower() for p in target_proposals)):
            continue

        proposal_id = app_dir.name

        original_files = {}
        for md_file in app_dir.glob("*.md"):
            original_files[md_file.stem] = md_file.read_text().strip()

        original_reviews = None
        reviews_file = app_dir / "original_reviews_saved.json"
        if reviews_file.exists():
            reviews_data = json.loads(reviews_file.read_text())
            original_reviews = [Review(**r) for r in reviews_data]

        perturbations = []
        perturbed_dir = app_dir / "perturbed_examples"
        if perturbed_dir.exists():
            perturbations = _load_perturbations(perturbed_dir, original_files)

        proposals.append(ProposalData(
            id=proposal_id,
            original_files=original_files,
            original_reviews=original_reviews,
            perturbations=perturbations
        ))

    return proposals


def _load_perturbations(base_dir: Path, original_files: dict[str, str]) -> list[PerturbationData]:
    perturbations = []

    for doc_file in base_dir.rglob("doc.txt"):
        perturb_dir = doc_file.parent
        relative_path = perturb_dir.relative_to(base_dir)
        perturb_name = str(relative_path).replace("/", ".")

        description = doc_file.read_text().strip()

        files = {}
        diffs = {}
        for md_file in perturb_dir.glob("*.md"):
            key = md_file.stem
            content = md_file.read_text().strip()

            if key in original_files and content.lower() != original_files[key].lower():
                files[key] = content
                diffs[key] = get_file_diff(original_files[key], content, key)

        if files:
            perturbations.append(PerturbationData(
                name=perturb_name,
                description=description,
                files=files,
                diffs=diffs
            ))

    return perturbations


def create_stable_id(*fields: Any, prefix: str = "", length: int = 12) -> str:
    combined = "\0".join(str(field) for field in fields)
    hash_value = hashlib.md5(combined.encode()).hexdigest()[:length]
    return f"{prefix}_{hash_value}" if prefix else hash_value


def get_file_diff(original: str, perturbed: str, filename: str, context_lines: int = 3, max_tokens: int = 2000) -> str:
    og_lines = original.splitlines()
    pt_lines = perturbed.splitlines()

    diff_lines = list(unified_diff(
        og_lines, pt_lines,
        fromfile=f"{filename}_original",
        tofile=f"{filename}_perturbed",
        n=context_lines,
        lineterm=''
    ))

    if len(diff_lines) <= 2:
        return ""

    full_diff = "\n".join(diff_lines)

    if len(full_diff) <= max_tokens * 4:
        return full_diff

    truncated_lines = []
    char_count = 0
    max_chars = max_tokens * 4

    for line in diff_lines:
        if char_count + len(line) > max_chars:
            if line.startswith('@@'):
                break
            if truncated_lines:
                break
        truncated_lines.append(line)
        char_count += len(line) + 1

    return "\n".join(truncated_lines) + "\n\n[diff truncated]"


def convert_to_grouped_sections(files: dict[str, str], return_opportunity=False, return_summary=False) -> dict[str, str]:
    # TODO - Add parsing for fit-to-opportunity file
    app_overview = parse_application_overview(
        files.get("application-overview", "")) if "application-overview" in files else None
    vision_approach = parse_vision_and_approach(
        files.get("vision-and-approach", "")) if "vision-and-approach" in files else None

    groups = {section_name: _build_section_content(app_overview, vision_approach, section_name)
                for section_name in grouped_sections}

    if return_opportunity:
        opportunity = parse_opportunity(
            files.get("opportunity", "")) if "opportunity" in files else None
        if opportunity:
            groups |= {"opportunity": opportunity}

    if return_summary and vision_approach:
        groups |= {"summary": app_overview.summary}

    return groups


def _build_full_prompt(files: dict[str, str]) -> str:
    sections = []
    if "opportunity" in files:
        sections.append(f"## Funding Opportunity\n{files['opportunity']}")
    if "application-overview" in files:
        sections.append(f"## Application Overview\n{files['application-overview']}")
    if "vision-and-approach" in files:
        sections.append(f"## Vision and Approach\n{files['vision-and-approach']}")
    if "fit-to-opportunity" in files:
        sections.append(f"## Fit to Opportunity\n{files['fit-to-opportunity']}")
    return "\n\n".join(sections)


def create_dataset(include_original: bool = True, include_perturbed: bool = True, target_proposals: str = ""):
    if target_proposals:
        target_proposals = target_proposals.split(",")
    else:
        target_proposals = []

    samples = []
    proposals = load_all_proposals(target_proposals)

    for proposal in proposals:
        if include_original:
            original_metadata = {
                "proposal_id": proposal.id,
                "sample_type": "original",
                "perturbation_name": None,
                "perturbation_description": None,
                "files": proposal.original_files,
            }

            if proposal.original_reviews:
                original_metadata["original_scores"] = [r.score for r in proposal.original_reviews]
                original_metadata["original_explanations"] = [r.explanation for r in proposal.original_reviews]

            sample_id = create_stable_id(proposal.id, "original", prefix="original")

            samples.append(Sample(
                id=sample_id,
                input=_build_full_prompt(proposal.original_files),
                metadata=original_metadata,
            ))

        if include_perturbed:
            for perturbation in proposal.perturbations:
                merged_files = proposal.original_files.copy()
                merged_files.update(perturbation.files)

                diffs = {}
                for filename in perturbation.files:
                    if filename in proposal.original_files:
                        diff = get_file_diff(
                            proposal.original_files[filename],
                            perturbation.files[filename],
                            filename
                        )
                        if diff:
                            diffs[filename] = diff

                if not diffs:
                    warnings.warn(f"No diff for any files of proposal '{proposal.id}' with perturbation '{perturbation.name}'. Skipping...")
                    continue

                perturb_metadata = {
                    "proposal_id": proposal.id,
                    "sample_type": "perturbed",
                    "perturbation_name": perturbation.name,
                    "perturbation_description": perturbation.description,
                    "perturbed_files": perturbation.files,
                    "files": merged_files,
                    "diffs": diffs,
                }

                sample_id = create_stable_id(proposal.id, perturbation.name, prefix="perturb")

                samples.append(Sample(
                    id=sample_id,
                    input=_build_full_prompt(merged_files),
                    metadata=perturb_metadata,
                ))

    return samples