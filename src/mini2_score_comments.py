from pathlib import Path
import re
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LABMT_PATH = ROOT / "data" / "clean" / "labMT_clean.csv"
COMMENTS_PATH = ROOT / "data" / "processed" / "civil_comments_sample.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "civil_comments_scored.csv"


def tokenize(text):
    """
    Convert comment text into lowercase alphabetic word tokens.
    """
    return re.findall(r"[a-z]+", str(text).lower())


def score_comment(text, labmt_scores):
    tokens = tokenize(text)

    matched_scores = [
        labmt_scores[token]
        for token in tokens
        if token in labmt_scores
    ]

    token_count = len(tokens)
    matched_token_count = len(matched_scores)

    if matched_token_count == 0:
        mean_happiness = None
        coverage_ratio = 0
    else:
        mean_happiness = sum(matched_scores) / matched_token_count
        coverage_ratio = matched_token_count / token_count if token_count > 0 else 0

    return token_count, matched_token_count, coverage_ratio, mean_happiness


def main():
    print("Loading cleaned labMT lexicon...")
    labmt = pd.read_csv(LABMT_PATH)

    labmt_scores = dict(
        zip(labmt["word"], labmt["happiness_average"])
    )

    print("Loading Civil Comments sample...")
    comments = pd.read_csv(COMMENTS_PATH)

    token_counts = []
    matched_counts = []
    coverage_ratios = []
    mean_scores = []

    print("Scoring Civil Comments with labMT...")

    for i, text in enumerate(comments["text"]):
        token_count, matched_count, coverage_ratio, mean_score = score_comment(text, labmt_scores)

        token_counts.append(token_count)
        matched_counts.append(matched_count)
        coverage_ratios.append(coverage_ratio)
        mean_scores.append(mean_score)

        if (i + 1) % 1000 == 0:
            print(f"Scored {i + 1} comments...")

    comments["token_count"] = token_counts
    comments["matched_token_count"] = matched_counts
    comments["coverage_ratio"] = coverage_ratios
    comments["mean_happiness"] = mean_scores

    comments.to_csv(OUTPUT_PATH, index=False)

    print("\nScored Civil Comments dataset saved.")
    print("Shape:", comments.shape)
    print("Saved to:", OUTPUT_PATH)

    print("\nMean happiness by toxicity group:")
    print(comments.groupby("toxicity_group")["mean_happiness"].mean())

    print("\nMatched token count summary:")
    print(comments.groupby("toxicity_group")["matched_token_count"].describe())

    print("\nCoverage ratio summary:")
    print(comments.groupby("toxicity_group")["coverage_ratio"].describe())


if __name__ == "__main__":
    main()