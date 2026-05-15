from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "processed" / "civil_comments_scored.csv"

FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"

FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)


def bootstrap_mean_difference(low_scores, high_scores, n_bootstrap=5000, random_state=42):
    """
    Bootstrap the mean difference between low-toxicity and high-toxicity comments.
    Difference is calculated as: low toxicity mean - high toxicity mean.
    """
    rng = np.random.default_rng(random_state)

    low_scores = np.array(low_scores)
    high_scores = np.array(high_scores)

    differences = []

    for _ in range(n_bootstrap):
        low_sample = rng.choice(low_scores, size=len(low_scores), replace=True)
        high_sample = rng.choice(high_scores, size=len(high_scores), replace=True)
        differences.append(low_sample.mean() - high_sample.mean())

    return np.array(differences)


def shorten_text(text, max_chars=350):
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def main():
    print("Loading scored Civil Comments...")
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["mean_happiness"]).copy()

    low = df[df["toxicity_group"] == "low_toxicity"].copy()
    high = df[df["toxicity_group"] == "high_toxicity"].copy()

    low_scores = low["mean_happiness"]
    high_scores = high["mean_happiness"]

    low_mean = low_scores.mean()
    high_mean = high_scores.mean()
    observed_diff = low_mean - high_mean

    print("\nBasic result:")
    print(f"Low-toxicity mean happiness: {low_mean:.6f}")
    print(f"High-toxicity mean happiness: {high_mean:.6f}")
    print(f"Observed mean difference, low - high: {observed_diff:.6f}")

    # ----------------------------
    # Table 2: Summary by toxicity group
    # ----------------------------
    summary = (
        df.groupby("toxicity_group")
        .agg(
            count=("comment_id", "count"),
            mean_toxicity=("toxicity", "mean"),
            mean_happiness=("mean_happiness", "mean"),
            median_happiness=("mean_happiness", "median"),
            std_happiness=("mean_happiness", "std"),
            mean_word_count=("word_count", "mean"),
            mean_matched_token_count=("matched_token_count", "mean"),
            mean_coverage_ratio=("coverage_ratio", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(TABLE_DIR / "mini2_table2_toxicity_summary.csv", index=False)

    # ----------------------------
    # Figure 7: Civil Comments scoring workflow
    # ----------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")

    steps = [
        "Civil Comments corpus\nonline comments with toxicity scores",
        "Select two balanced groups\nlow toxicity <= 0.10 and high toxicity >= 0.80",
        "Create processed sample\n5,000 low-toxicity + 5,000 high-toxicity comments",
        "Tokenize comment text\nlowercase alphabetic word tokens",
        "Match tokens to cleaned labMT lexicon\nfrom data/clean/labMT_clean.csv",
        "Calculate comment-level scores\nmatched_token_count, coverage_ratio, mean_happiness",
        "Compare emotional valence and toxicity\nfigures, tables, bootstrap, mismatch examples",
    ]

    y_positions = np.linspace(0.88, 0.10, len(steps))

    for i, (step, y) in enumerate(zip(steps, y_positions), start=1):
        ax.text(
            0.5,
            y,
            f"{i}. {step}",
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="white"),
        )

        if i < len(steps):
            ax.annotate(
                "",
                xy=(0.5, y_positions[i] + 0.045),
                xytext=(0.5, y - 0.045),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

    ax.set_title("Civil Comments scoring workflow with labMT", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mini2_fig7_civil_comments_workflow.png", dpi=300)
    plt.close()

    # ----------------------------
    # Figure 8: Distribution by toxicity group
    # ----------------------------
    plt.figure(figsize=(10, 6))

    plt.hist(
        high_scores,
        bins=45,
        density=True,
        alpha=0.6,
        label=f"High toxicity, mean = {high_mean:.3f}",
        edgecolor="black",
    )

    plt.hist(
        low_scores,
        bins=45,
        density=True,
        alpha=0.6,
        label=f"Low toxicity, mean = {low_mean:.3f}",
        edgecolor="black",
    )

    plt.axvline(high_mean, linestyle="--", linewidth=2)
    plt.axvline(low_mean, linestyle="-", linewidth=2)

    plt.title("Distribution of labMT happiness scores by toxicity group")
    plt.xlabel("Comment-level mean happiness score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mini2_fig8_happiness_distribution_by_toxicity.png", dpi=300)
    plt.close()

    # ----------------------------
    # Figure 9: Boxplot by toxicity group
    # ----------------------------
    plt.figure(figsize=(8, 6))

    plt.boxplot(
        [high_scores, low_scores],
        tick_labels=["High toxicity", "Low toxicity"],
        showfliers=False,
    )

    plt.title("Comment-level happiness scores by toxicity group")
    plt.xlabel("Toxicity group")
    plt.ylabel("Mean labMT happiness score")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mini2_fig9_boxplot_by_toxicity.png", dpi=300)
    plt.close()

    # ----------------------------
    # Figure 10: Bootstrap mean difference
    # ----------------------------
    print("\nRunning bootstrap...")
    boot_diffs = bootstrap_mean_difference(low_scores, high_scores)

    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)

    bootstrap_summary = pd.DataFrame(
        {
            "metric": [
                "low_toxicity_mean",
                "high_toxicity_mean",
                "observed_difference_low_minus_high",
                "bootstrap_ci_low_2.5",
                "bootstrap_ci_high_97.5",
            ],
            "value": [
                low_mean,
                high_mean,
                observed_diff,
                ci_low,
                ci_high,
            ],
        }
    )

    bootstrap_summary.to_csv(TABLE_DIR / "mini2_table3_bootstrap_summary.csv", index=False)

    plt.figure(figsize=(10, 6))

    plt.hist(boot_diffs, bins=45, edgecolor="black", alpha=0.8)
    plt.axvline(observed_diff, linewidth=2, label=f"Observed diff = {observed_diff:.3f}")
    plt.axvline(ci_low, linestyle="--", linewidth=2, label=f"95% CI low = {ci_low:.3f}")
    plt.axvline(ci_high, linestyle="--", linewidth=2, label=f"95% CI high = {ci_high:.3f}")
    plt.axvline(0, linestyle=":", linewidth=2, label="Zero difference")

    plt.title("Bootstrap distribution of mean happiness difference")
    plt.xlabel("Mean difference: low toxicity - high toxicity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mini2_fig10_bootstrap_mean_difference.png", dpi=300)
    plt.close()

    # ----------------------------
    # Figure 11: Small-sample stability check
    # ----------------------------
    rng = np.random.default_rng(42)

    sample_size = 500
    n_repeats = 500
    small_sample_diffs = []

    for _ in range(n_repeats):
        low_sample = rng.choice(low_scores, size=sample_size, replace=False)
        high_sample = rng.choice(high_scores, size=sample_size, replace=False)
        diff = low_sample.mean() - high_sample.mean()
        small_sample_diffs.append(diff)

    small_sample_diffs = np.array(small_sample_diffs)
    proportion_positive = (small_sample_diffs > 0).mean()

    small_sample_summary = pd.DataFrame({
        "metric": [
            "sample_size_per_group",
            "number_of_repeats",
            "mean_small_sample_difference",
            "min_difference",
            "max_difference",
            "proportion_of_samples_with_positive_difference"
        ],
        "value": [
            sample_size,
            n_repeats,
            small_sample_diffs.mean(),
            small_sample_diffs.min(),
            small_sample_diffs.max(),
            proportion_positive
        ]
    })

    small_sample_summary.to_csv(
        TABLE_DIR / "mini2_table4_small_sample_stability.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))
    plt.hist(small_sample_diffs, bins=35, edgecolor="black", alpha=0.8)
    plt.axvline(0, linestyle=":", linewidth=2, label="Zero difference")
    plt.axvline(
        small_sample_diffs.mean(),
        linewidth=2,
        label=f"Mean small-sample diff = {small_sample_diffs.mean():.3f}"
    )

    plt.title("Small-sample stability of the happiness difference")
    plt.xlabel("Mean difference: low toxicity - high toxicity")
    plt.ylabel("Frequency across repeated samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mini2_fig11_small_sample_stability.png", dpi=300)
    plt.close()

    # ----------------------------
    # Figure 12: Matched token count by toxicity group
    # ----------------------------
    plt.figure(figsize=(10, 6))

    plt.hist(
        high["matched_token_count"],
        bins=50,
        alpha=0.6,
        label="High toxicity",
        edgecolor="black",
    )

    plt.hist(
        low["matched_token_count"],
        bins=50,
        alpha=0.6,
        label="Low toxicity",
        edgecolor="black",
    )

    plt.title("Matched labMT token count by toxicity group")
    plt.xlabel("Matched token count")
    plt.ylabel("Number of comments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mini2_fig12_matched_token_count_by_toxicity.png", dpi=300)
    plt.close()

    # ----------------------------
    # Figure 13: Toxicity score vs happiness
    # ----------------------------
    sample_for_scatter = df.sample(n=min(8000, len(df)), random_state=42)

    corr_toxicity_happiness = sample_for_scatter["toxicity"].corr(
        sample_for_scatter["mean_happiness"]
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(
        sample_for_scatter["toxicity"],
        sample_for_scatter["mean_happiness"],
        alpha=0.35,
        s=12,
    )

    plt.title("Toxicity score and labMT happiness score")
    plt.xlabel("Toxicity score")
    plt.ylabel("Mean labMT happiness score")

    plt.text(
        0.03,
        0.95,
        f"Correlation = {corr_toxicity_happiness:.2f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "mini2_fig13_toxicity_vs_happiness.png", dpi=300)
    plt.close()

    # ----------------------------
    # Figure 14: Coverage ratio by toxicity group
    # ----------------------------
    plt.figure(figsize=(8, 6))

    plt.boxplot(
        [high["coverage_ratio"], low["coverage_ratio"]],
        tick_labels=["High toxicity", "Low toxicity"],
        showfliers=False,
    )

    plt.title("labMT coverage ratio by toxicity group")
    plt.xlabel("Toxicity group")
    plt.ylabel("Coverage ratio")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mini2_fig14_coverage_ratio_by_toxicity.png", dpi=300)
    plt.close()

    # ----------------------------
    # Table 5: Mismatch examples
    # ----------------------------
    # High-toxicity comments with unusually high happiness
    high_high_happiness = high.sort_values("mean_happiness", ascending=False).head(5).copy()
    high_high_happiness["mismatch_type"] = "high_toxicity_high_labMT_happiness"

    # Low-toxicity comments with unusually low happiness
    low_low_happiness = low.sort_values("mean_happiness", ascending=True).head(5).copy()
    low_low_happiness["mismatch_type"] = "low_toxicity_low_labMT_happiness"

    mismatch = pd.concat([high_high_happiness, low_low_happiness], ignore_index=True)
    mismatch["short_text"] = mismatch["text"].apply(lambda x: shorten_text(x, max_chars=400))

    mismatch_table = mismatch[
        [
            "mismatch_type",
            "comment_id",
            "toxicity_group",
            "toxicity",
            "word_count",
            "matched_token_count",
            "coverage_ratio",
            "mean_happiness",
            "short_text",
        ]
    ]

    mismatch_table.to_csv(TABLE_DIR / "mini2_table5_mismatch_examples.csv", index=False)

    with open(TABLE_DIR / "mini2_mismatch_examples_readable.txt", "w", encoding="utf-8") as f:
        for _, row in mismatch_table.iterrows():
            f.write("=" * 80 + "\n")
            f.write(f"Mismatch type: {row['mismatch_type']}\n")
            f.write(f"Comment ID: {row['comment_id']}\n")
            f.write(f"Toxicity group: {row['toxicity_group']}\n")
            f.write(f"Toxicity score: {row['toxicity']:.3f}\n")
            f.write(f"Mean happiness: {row['mean_happiness']:.3f}\n")
            f.write(f"Coverage ratio: {row['coverage_ratio']:.3f}\n")
            f.write(f"Text: {row['short_text']}\n\n")

    print("\nPart 2 figures saved to figures/")
    print("Part 2 tables saved to tables/")

    print("\nKey results:")
    print(f"Low-toxicity mean happiness: {low_mean:.6f}")
    print(f"High-toxicity mean happiness: {high_mean:.6f}")
    print(f"Observed difference, low - high: {observed_diff:.6f}")
    print(f"Bootstrap 95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Small-sample mean difference: {small_sample_diffs.mean():.6f}")
    print(f"Proportion of positive small-sample differences: {proportion_positive:.3f}")
    print(f"Toxicity vs happiness correlation: {corr_toxicity_happiness:.3f}")


if __name__ == "__main__":
    main()