from pathlib import Path
import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "processed" / "civil_comments_sample.csv"

SAMPLE_PER_GROUP = 5000
MAX_EXAMPLES_TO_SCAN = 300000

LOW_TOXICITY_THRESHOLD = 0.10
HIGH_TOXICITY_THRESHOLD = 0.80


def main():
    print("Loading Civil Comments from Hugging Face datasets...")
    print("This can take a while the first time.")

    dataset = load_dataset("civil_comments", split="train", streaming=True)

    low_rows = []
    high_rows = []

    scanned = 0

    for example in dataset:
        scanned += 1

        text = str(example["text"])
        toxicity = float(example["toxicity"])

        if toxicity <= LOW_TOXICITY_THRESHOLD and len(low_rows) < SAMPLE_PER_GROUP:
            low_rows.append({
                "comment_id": f"low_{len(low_rows)}",
                "text": text,
                "toxicity": toxicity,
                "toxicity_group": "low_toxicity",
            })

        elif toxicity >= HIGH_TOXICITY_THRESHOLD and len(high_rows) < SAMPLE_PER_GROUP:
            high_rows.append({
                "comment_id": f"high_{len(high_rows)}",
                "text": text,
                "toxicity": toxicity,
                "toxicity_group": "high_toxicity",
            })

        if scanned % 10000 == 0:
            print(
                f"Scanned {scanned} comments | "
                f"low: {len(low_rows)} | high: {len(high_rows)}"
            )

        if len(low_rows) >= SAMPLE_PER_GROUP and len(high_rows) >= SAMPLE_PER_GROUP:
            break

        if scanned >= MAX_EXAMPLES_TO_SCAN:
            print("Reached maximum scan limit.")
            break

    df = pd.DataFrame(low_rows + high_rows)
    df["word_count"] = df["text"].astype(str).str.split().str.len()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nCivil Comments sample saved.")
    print("Saved to:", OUTPUT_PATH)
    print("Shape:", df.shape)

    print("\nGroup counts:")
    print(df["toxicity_group"].value_counts())

    print("\nToxicity summary:")
    print(df.groupby("toxicity_group")["toxicity"].describe())

    print("\nWord count summary:")
    print(df.groupby("toxicity_group")["word_count"].describe())


if __name__ == "__main__":
    main()