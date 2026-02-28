import os
import json

# ====== CONFIGURATION ======
INPUT_FOLDERS = [
    "ST4_Complex_NonSymbolic_Final",
    "ST4_Complex_Symbolic_Final",
    "ST4_Simple_All_Translated",
    "ST4_Simple_Symbolic_All_Translated"
]

LANGUAGES = [
    "english",
    "german",
    "spanish",
    "french",
    "italian",
    "dutch",
    "portuguese",
    "russian",
    "chinese",
    "swahili",
    "bengali",
    "telugu"
]

OUTPUT_FOLDER = "ST4_Final"
# ===============================


def combine_language_files():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for language in LANGUAGES:
        combined_data = []
        print(f"\nProcessing language: {language}")

        for folder in INPUT_FOLDERS:
            if not os.path.isdir(folder):
                print(f"Skipping missing folder: {folder}")
                continue

            for filename in os.listdir(folder):
                # Match language name in filename and ensure it's JSON
                if language.lower() in filename.lower() and filename.endswith(".json"):
                    file_path = os.path.join(folder, filename)

                    print(f"  Adding file: {file_path}")

                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        if isinstance(data, list):
                            combined_data.extend(data)
                        else:
                            print(f"  Skipped (not a list): {file_path}")

        # Write combined JSON file
        if combined_data:
            output_path = os.path.join(
                OUTPUT_FOLDER, f"{language}_precomplex_symbolic.json"
            )

            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(combined_data, out_f, ensure_ascii=False, indent=4)

            print(f"  Created: {output_path} ({len(combined_data)} datapoints)")
        else:
            print(f"  No valid JSON files found for: {language}")


if __name__ == "__main__":
    combine_language_files()