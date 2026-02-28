import json
from pathlib import Path

# ---------------- CONFIG ---------------- #

LANGUAGES = [
    "bengali", "french", "german", "italian", "dutch",
    "portuguese", "russian", "chinese", "swahili", "telugu", "spanish"
]

# ---------------------------------------- #

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def build_id_map(data):
    """
    Converts list of datapoints into dict keyed by id
    """
    return {item["id"]: item for item in data}

def main():
    base_dir = Path(__file__).parent
    simple_dir = base_dir / "simple_symbolic_4_var"
    complex_dir = base_dir / "complex_symbolic_4_var"
    output_path = base_dir / "st3_kl_symbolic_4_var_data.json"

    # ---------------- Load English First ---------------- #

    english_simple = load_json(simple_dir / "english_4_var_simple_symbolic.json")
    english_complex = load_json(complex_dir / "english_4_var_complex_symbolic.json")

    eng_simple_map = build_id_map(english_simple)
    eng_complex_map = build_id_map(english_complex)

    final_dataset = []

    print("Pairing datasets...")

    # Loop over languages (excluding English)
    for language in LANGUAGES:

        print(f"Processing {language}...")

        lang_simple_path = simple_dir / f"{language}_4_var_simple_symbolic.json"
        lang_complex_path = complex_dir / f"{language}_4_var_complex_symbolic.json"

        if not lang_simple_path.exists() or not lang_complex_path.exists():
            print(f"Skipping {language}, files not found.")
            continue

        lang_simple = load_json(lang_simple_path)
        lang_complex = load_json(lang_complex_path)

        lang_simple_map = build_id_map(lang_simple)
        lang_complex_map = build_id_map(lang_complex)

        # Iterate over English IDs
        for id_ in eng_simple_map:

            if (
                id_ not in eng_complex_map
                or id_ not in lang_simple_map
                or id_ not in lang_complex_map
            ):
                continue  # skip if any missing

            eng_s = eng_simple_map[id_]
            eng_c = eng_complex_map[id_]
            lang_s = lang_simple_map[id_]
            lang_c = lang_complex_map[id_]

            datapoint = {
                "id": id_,
                "validity": eng_s.get("validity"),
                "plausibility": eng_s.get("plausibility"),
                "language": language,

                "english_simple": eng_s.get("syllogism"),
                "english_complex": eng_c.get("syllogism"),

                f"{language}_simple": lang_s.get("syllogism"),
                f"{language}_complex": lang_c.get("syllogism"),
            }

            final_dataset.append(datapoint)

    save_json(final_dataset, output_path)

    print(f"\nDone! Saved dataset to {output_path}")
    print(f"Total datapoints: {len(final_dataset)}")

if __name__ == "__main__":
    main()
