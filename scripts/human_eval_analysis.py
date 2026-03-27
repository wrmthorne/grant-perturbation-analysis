from pathlib import Path
import json
import pandas as pd
import numpy as np
from itertools import combinations
from scipy import stats



ROOT_PATH = Path(__file__).parent
SAVE_ROOT = ROOT_PATH / "ANNOTATION_ROUND_2"
SAMPLES_DIR = SAVE_ROOT / "annotation_samples"


# Load all annotations
def load_annotations():
    all_annotations = []
    for letter in "ABCD":
        path = SAMPLES_DIR / f"complete_annotation_data_{letter}.json"
        if path.exists():
            data = json.loads(path.read_text())
            for row in data:
                row["annotator_group"] = letter
            all_annotations.extend(data)
    return pd.DataFrame(all_annotations)


# Load judged data for source/system info
def load_judged_data():
    return pd.read_json(SAVE_ROOT / "all_judged_with_idx.json")


annotations = load_annotations()
judged = load_judged_data()

# Align: comment_id in annotations should match global_idx in judged
annotations = annotations.merge(
    judged[["global_idx", "source", "category"]].drop_duplicates(),
    left_on="comment_id", right_on="global_idx", how="left"
)

# Determine if claim is LLM-only or Human-only based on filename pattern
# Files like "judged_llm_to_human_*" contain LLM claims not in human reviews
annotations["claim_origin"] = annotations["filename"].apply(
    lambda x: "LLM" if "llm_to_human" in str(x) else "Human" if "human_to_llm" in str(x) else "Unknown"
)


# Extract review system from source column (baseline, sectioned, council, or human)
def extract_system(source):
    if pd.isna(source):
        return "unknown"
    s = str(source).lower()
    if "baseline" in s:
        return "baseline"
    elif "sectioned" in s:
        return "sectioned"
    elif "council" in s:
        return "council"
    elif "human" in s or "expert" in s:
        return "human"
    return source


annotations["review_system"] = annotations["source"].apply(extract_system)

# Convert validity to binary
annotations["valid"] = annotations["validity"].str.lower() == "yes"

# Ordinal mappings for agreement analysis
agreement_map = {"strongly disagree": 1, "disagree": 2, "neutral": 3, "agree": 4, "strongly agree": 5}
impact_map = {"minimal": 1, "moderate": 2, "substantial": 3}
annotations["agreement_ord"] = annotations["agreement"].str.lower().map(agreement_map)
annotations["impact_ord"] = annotations["impact"].str.lower().map(impact_map)


# === INTER-ANNOTATOR AGREEMENT ===

def cohens_kappa(y1, y2):
    """Compute Cohen's Kappa for two raters."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y1, y2)


def krippendorff_alpha_nominal(data_matrix):
    """Simplified Krippendorff's alpha for nominal data (validity)."""
    # data_matrix: rows=items, cols=annotators, NaN for missing
    n_items, n_raters = data_matrix.shape

    # Observed disagreement
    observed = 0
    n_pairs = 0
    for i in range(n_items):
        ratings = data_matrix[i, ~np.isnan(data_matrix[i, :])]
        m = len(ratings)
        if m < 2:
            continue
        for j in range(m):
            for k in range(j + 1, m):
                observed += (ratings[j] != ratings[k])
                n_pairs += 1

    if n_pairs == 0:
        return np.nan
    observed /= n_pairs

    # Expected disagreement
    all_ratings = data_matrix[~np.isnan(data_matrix)]
    vals, counts = np.unique(all_ratings, return_counts=True)
    probs = counts / counts.sum()
    expected = 1 - np.sum(probs ** 2)

    if expected == 0:
        return 1.0
    return 1 - observed / expected


# Find overlapping annotations (same comment_id, different user_id)
def compute_iaa(annotations):
    # Group by comment_id to find items with multiple annotators
    grouped = annotations.groupby("comment_id")

    overlap_items = []
    for comment_id, group in grouped:
        users = group["user_id"].unique()
        if len(users) >= 2:
            overlap_items.append(comment_id)

    print(f"Items with 2+ annotators: {len(overlap_items)}")

    if len(overlap_items) == 0:
        print("No overlapping annotations found for IAA computation")
        return {}

    overlap_df = annotations[annotations["comment_id"].isin(overlap_items)]

    # Build matrix for Krippendorff's alpha
    users = overlap_df["user_id"].unique()
    user_to_idx = {u: i for i, u in enumerate(users)}

    validity_matrix = np.full((len(overlap_items), len(users)), np.nan)
    agreement_matrix = np.full((len(overlap_items), len(users)), np.nan)
    impact_matrix = np.full((len(overlap_items), len(users)), np.nan)

    item_to_idx = {c: i for i, c in enumerate(overlap_items)}

    for _, row in overlap_df.iterrows():
        i = item_to_idx[row["comment_id"]]
        j = user_to_idx[row["user_id"]]
        validity_matrix[i, j] = float(row["valid"])
        if pd.notna(row["agreement_ord"]):
            agreement_matrix[i, j] = row["agreement_ord"]
        if pd.notna(row["impact_ord"]):
            impact_matrix[i, j] = row["impact_ord"]

    results = {
        "n_overlap_items": len(overlap_items),
        "n_annotators": len(users),
        "validity_alpha": krippendorff_alpha_nominal(validity_matrix),
    }

    # Pairwise Cohen's Kappa for users with sufficient overlap
    kappas = []
    for u1, u2 in combinations(users, 2):
        mask = ~np.isnan(validity_matrix[:, user_to_idx[u1]]) & ~np.isnan(validity_matrix[:, user_to_idx[u2]])
        if mask.sum() >= 5:
            k = cohens_kappa(
                validity_matrix[mask, user_to_idx[u1]],
                validity_matrix[mask, user_to_idx[u2]]
            )
            kappas.append({"user1": u1, "user2": u2, "kappa": k, "n": mask.sum()})

    results["pairwise_kappas"] = kappas
    return results


iaa_results = compute_iaa(annotations)
print("\n=== Inter-Annotator Agreement ===")
print(f"Krippendorff's alpha (validity): {iaa_results.get('validity_alpha', 'N/A'):.3f}")
if iaa_results.get("pairwise_kappas"):
    for kp in iaa_results["pairwise_kappas"]:
        print(f"  Cohen's Kappa ({kp['user1'][:8]}... vs {kp['user2'][:8]}...): {kp['kappa']:.3f} (n={kp['n']})")

# === RESEARCH QUESTIONS ===

print("\n=== Q1: Valid claims unique to LLM vs Human ===")

# For claims that LLM made but humans didn't (llm_to_human files)
llm_unique = annotations[annotations["claim_origin"] == "LLM"]
human_unique = annotations[annotations["claim_origin"] == "Human"]


# Aggregate validity across annotators (majority vote or any valid)
def aggregate_validity(df, method="majority"):
    grouped = df.groupby("comment_id")["valid"].agg(["sum", "count"])
    if method == "majority":
        return (grouped["sum"] > grouped["count"] / 2).sum(), len(grouped)
    elif method == "any":
        return (grouped["sum"] > 0).sum(), len(grouped)


# Per-system breakdown for LLM claims
print("\nLLM claims not made by humans (by review system):")
for system in ["baseline", "sectioned", "council"]:
    subset = llm_unique[llm_unique["review_system"] == system]
    if len(subset) > 0:
        valid_count, total = aggregate_validity(subset)
        print(f"  {system}: {valid_count}/{total} valid ({100 * valid_count / total:.1f}%)")

print("\nHuman claims not made by LLMs:")
valid_count, total = aggregate_validity(human_unique)
print(f"  Total: {valid_count}/{total} valid ({100 * valid_count / total:.1f}%)")

print("\n=== Q2: Total valid claims ===")

# All claims by origin
for origin in ["LLM", "Human"]:
    subset = annotations[annotations["claim_origin"] == origin]
    valid_count, total = aggregate_validity(subset)
    print(f"{origin}-unique claims: {valid_count}/{total} valid ({100 * valid_count / total:.1f}%)")

print("\n=== Q3: Impact distribution of valid claims ===")

valid_annotations = annotations[annotations["valid"]]

print("\nImpact distribution (LLM-unique valid claims):")
llm_valid = valid_annotations[valid_annotations["claim_origin"] == "LLM"]
if len(llm_valid) > 0:
    print(llm_valid["impact"].value_counts(normalize=True).round(3))

print("\nImpact distribution (Human-unique valid claims):")
human_valid = valid_annotations[valid_annotations["claim_origin"] == "Human"]
if len(human_valid) > 0:
    print(human_valid["impact"].value_counts(normalize=True).round(3))

# Chi-square test for impact distribution difference
if len(llm_valid) > 0 and len(human_valid) > 0:
    contingency = pd.crosstab(valid_annotations["claim_origin"], valid_annotations["impact"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-square test (impact ~ origin): χ²={chi2:.2f}, p={p:.4f}")

print("\n=== Additional: Validity by category ===")
for cat in annotations["category"].dropna().unique():
    subset = annotations[annotations["category"] == cat]
    valid_count, total = aggregate_validity(subset)
    if total > 0:
        print(f"  {cat}: {valid_count}/{total} valid ({100 * valid_count / total:.1f}%)")

print("\n=== Additional: Validity by review system (LLM only) ===")
llm_only = annotations[annotations["claim_origin"] == "LLM"]
systems = llm_only["review_system"].unique()
validity_by_system = {}
for sys in systems:
    subset = llm_only[llm_only["review_system"] == sys]
    valid_count, total = aggregate_validity(subset)
    validity_by_system[sys] = (valid_count, total)
    print(f"  {sys}: {valid_count}/{total} ({100 * valid_count / total:.1f}%)")

# Chi-square for validity differences across systems
if len(systems) > 1:
    contingency = pd.crosstab(llm_only["review_system"], llm_only["valid"])
    if contingency.shape[1] == 2:  # Both True and False present
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-square (validity ~ system): χ²={chi2:.2f}, p={p:.4f}")

print("\n=== Additional: Correlation validity-impact ===")
# Point-biserial correlation (binary validity vs ordinal impact)
valid_mask = annotations["impact_ord"].notna()
if valid_mask.sum() > 10:
    r, p = stats.pointbiserialr(
        annotations.loc[valid_mask, "valid"].astype(int),
        annotations.loc[valid_mask, "impact_ord"]
    )
    print(f"Point-biserial r (valid vs impact): r={r:.3f}, p={p:.4f}")

# === SUMMARY TABLE ===
print("\n=== Summary Table ===")
summary_data = []
for origin in ["LLM", "Human"]:
    subset = annotations[annotations["claim_origin"] == origin]
    valid_count, total = aggregate_validity(subset)
    valid_subset = subset[subset["valid"]]

    impact_dist = valid_subset["impact"].value_counts().to_dict() if len(valid_subset) > 0 else {}

    summary_data.append({
        "Origin": origin,
        "Total Claims": total,
        "Valid Claims": valid_count,
        "Validity Rate": f"{100 * valid_count / total:.1f}%" if total > 0 else "N/A",
        "Minimal Impact": impact_dist.get("minimal", 0),
        "Moderate Impact": impact_dist.get("moderate", 0),
        "Substantial Impact": impact_dist.get("substantial", 0),
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save results
results_output = {
    "iaa": iaa_results,
    "summary": summary_data,
}
with open(SAVE_ROOT / "analysis_results.json", "w") as f:
    json.dump(results_output, f, indent=2, default=str)
print(f"\n✓ Results saved to {SAVE_ROOT / 'analysis_results.json'}")