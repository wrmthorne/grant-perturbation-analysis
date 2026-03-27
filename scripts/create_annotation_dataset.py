from pathlib import Path
import json
import pandas as pd

from inspect_utils import _build_section_content, parse_application_overview, parse_vision_and_approach



ROOT_PATH = Path(__file__).parent
SAVE_ROOT = ROOT_PATH / "ANNOTATION_ROUND_2"
CLAIMS_DIR = SAVE_ROOT / "final_claims"

apps = ["APP_A", "APP_B"]
section_types = ["vision_approach", "team_capability", "funding_resources", "ethics"]
section_names = {
    "vision_approach": "Vision and Approach",
    "team_capability": "Team Capability",
    "funding_resources": "Funding Resources",
    "ethics": "Ethics"
}

mapping = {
    "QUALITY/CLARITY": "vision_approach",
    "IMPACT": "vision_approach",
    "FUNDING": "funding_resources",
    "COMPETENCY": "team_capability",
    "ETHICS": "ethics",
    "ALIGNMENT": "vision_approach",
    "TIMELINE": "vision_approach",
}

annotator_assignments = {
    "A": [("APP_A", "vision_approach"), ("APP_A", "funding_resources"), ("APP_B", "ethics")],
    "B": [("APP_A", "vision_approach"), ("APP_B", "team_capability"), ("APP_B", "ethics")],
    "C": [("APP_A", "team_capability"), ("APP_B", "vision_approach"), ("APP_B", "funding_resources")],
    "D": [("APP_A", "funding_resources"), ("APP_A", "ethics"), ("APP_B", "team_capability")],
}

section_times = [
    {"section": 1, "durationMinutes": 20},
    {"section": 2, "durationMinutes": 20},
    {"section": 3, "durationMinutes": 10}
]


def load_and_prepare(judged_path, no_exact_path, proposal):
    judged = pd.read_json(judged_path).assign(proposal=proposal)
    no_exact = pd.read_json(no_exact_path).assign(proposal=proposal)

    # Create aligned IDs using original index position in judged file
    judged["global_idx"] = judged["query_idx"].astype(str) + "_" + judged["source"]

    # Match no_exact rows to judged by query_idx + source (their natural key)
    no_exact["global_idx"] = no_exact["query_idx"].astype(str) + "_" + no_exact["source"]

    return judged, no_exact


# Load all files
judged_app_B_lh, no_exact_app_B_lh = load_and_prepare(
    CLAIMS_DIR / "judged_llm_to_human_app_B.json", CLAIMS_DIR / "judged_llm_to_human_app_B_no_exact.json", "APP_B")
judged_app_B_hl, no_exact_app_B_hl = load_and_prepare(
    CLAIMS_DIR / "judged_human_to_llm_app_B.json", CLAIMS_DIR / "judged_human_to_llm_app_B_no_exact.json", "APP_B")
judged_app_A_lh, no_exact_app_A_lh = load_and_prepare(
    CLAIMS_DIR / "judged_llm_to_human_app_A.json", CLAIMS_DIR / "judged_llm_to_human_app_A_no_exact.json", "APP_A")
judged_app_A_hl, no_exact_app_A_hl = load_and_prepare(
    CLAIMS_DIR / "judged_human_to_llm_app_A.json", CLAIMS_DIR / "judged_human_to_llm_app_A_no_exact.json", "APP_A")

# Combine
all_judged = pd.concat([judged_app_B_lh, judged_app_B_hl, judged_app_A_lh, judged_app_A_hl])
all_no_exact = pd.concat([no_exact_app_B_lh, no_exact_app_B_hl, no_exact_app_A_lh, no_exact_app_A_hl])

all_judged["section"] = all_judged["category"].map(mapping)
all_no_exact["section"] = all_no_exact["category"].map(mapping)
all_no_exact["query"] = all_no_exact["query"].str.strip("-* ")

# Save the full judged data with global_idx for post-annotation joining
all_judged.to_json(SAVE_ROOT / "all_judged_with_idx.json", orient="records", indent=2)

# Load review guidelines
guidelines = json.loads((SAVE_ROOT / "review_guidelines.json").read_text())
guidelines_sections = {}
for section_type in section_types:
    md_parts = ["# Review Guidelines\n\n## Criteria\n\n"]
    for i, criterion in enumerate(guidelines[section_type]["criteria"], 1):
        md_parts.append(f"{i}. {criterion}\n\n")
    if "examples" in guidelines[section_type]:
        md_parts.append("## Examples\n\n")
        for i, example in enumerate(guidelines[section_type]["examples"], 1):
            md_parts.append(f"{i}. {example}\n\n")
    guidelines_sections[section_type] = "".join(md_parts)

# Pre-load proposal content
proposal_data = {}
for app in apps:
    app_path = ROOT_PATH / app
    proposal_data[app] = {
        "opportunity": (app_path / "opportunity.md").read_text(),
        "app_overview": parse_application_overview((app_path / "application-overview.md").read_text()),
        "v_and_a": parse_vision_and_approach((app_path / "vision-and-approach.md").read_text()),
    }

# Build per-annotator output files
for annotator, assignments in annotator_assignments.items():
    proposals = []

    for idx, (app, section_type) in enumerate(assignments):
        data = proposal_data[app]
        section_content = _build_section_content(data["app_overview"], data["v_and_a"], section_type)

        section_df = all_no_exact[(all_no_exact["proposal"] == app) & (all_no_exact["section"] == section_type)]
        comments = [
            {"id": row["global_idx"], "text": row["query"], "filename": row["filename"]}
            for _, row in section_df.iterrows()
        ]

        proposal_entry = next((p for p in proposals if p["id"] == app), None)
        if proposal_entry is None:
            proposal_entry = {"id": app, "sections": []}
            proposals.append(proposal_entry)

        proposal_entry["sections"].append({
            "id": section_type,
            "name": section_names[section_type],
            "opportunityDescription": data["opportunity"],
            "proposalContent": section_content,
            "reviewGuidelines": guidelines_sections[section_type],
            "comments": comments,
            "timedSection": idx
        })

    output = {"sectionOrder": section_times, "proposals": proposals}
    output_path = SAVE_ROOT / f"annotation_samples/pre_annotation_data_{annotator}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    total_comments = sum(len(s["comments"]) for p in proposals for s in p["sections"])
    print(f"✓ Annotator {annotator}: {output_path.name} ({total_comments} comments)")

print(f"\n✓ Saved full judged data with IDs to: {SAVE_ROOT / 'all_judged_with_idx.json'} ({len(all_judged)} records)")
print(f"✓ No-exact subset for annotation: {len(all_no_exact)} records")