import json
import random
import re
from pathlib import Path


mappings = [
    {"original": "mammals", "equivalent": "mammalian creatures"},
    {"original": "canines", "equivalent": "members of the dog family"},
    {"original": "felines", "equivalent": "members of the cat family"},
    {"original": "aquatic", "equivalent": "water-dwelling organisms"},
    {"original": "nocturnal", "equivalent": "active during the night"},
    {"original": "diurnal", "equivalent": "active during the day"},
    {"original": "toxic", "equivalent": "poisonous substances"},
    {"original": "lethal", "equivalent": "deadly elements"},
    {"original": "fragile", "equivalent": "easily broken objects"},
    {"original": "edible", "equivalent": "items fit for consumption"},
    {"original": "portable", "equivalent": "easily carried items"},
    {"original": "visible", "equivalent": "perceptible to the eye"},
    {"original": "audible", "equivalent": "capable of being heard"},
    {"original": "physicians", "equivalent": "medical doctors"},
    {"original": "attorneys", "equivalent": "legal professionals"},
    {"original": "instructors", "equivalent": "educational teachers"},
    {"original": "terrestrial", "equivalent": "land-based inhabitants"},
    {"original": "avian", "equivalent": "bird-like species"},
    {"original": "reptiles", "equivalent": "reptilian lifeforms"},
    {"original": "amphibians", "equivalent": "amphibious creatures"},
    {"original": "carnivores", "equivalent": "meat-eating animals"},
    {"original": "herbivores", "equivalent": "plant-eating animals"},
    {"original": "omnivores", "equivalent": "creatures with a varied diet"},
    {"original": "astronomers", "equivalent": "space scientists"},
    {"original": "botanists", "equivalent": "plant scientists"},
    {"original": "transparent", "equivalent": "see-through materials"},
    {"original": "opaque", "equivalent": "non-transparent objects"},
    {"original": "flammable", "equivalent": "highly combustible materials"},
    {"original": "metallic", "equivalent": "substances made of metal"},
    {"original": "organic", "equivalent": "carbon-based compounds"},
    {"original": "synthetic", "equivalent": "man-made materials"},
    {"original": "domestic", "equivalent": "tame household animals"},
    {"original": "predators", "equivalent": "natural hunters"},
    {"original": "prey", "equivalent": "hunted animals"},
    {"original": "microscopic", "equivalent": "tiny organisms"},
    {"original": "luminous", "equivalent": "light-emitting objects"},
    {"original": "stationary", "equivalent": "non-moving entities"},
    {"original": "vocal", "equivalent": "sound-producing creatures"},
    {"original": "subterranean", "equivalent": "underground dwellers"},
    {"original": "ancient", "equivalent": "very old artifacts"},
    {"original": "modern", "equivalent": "contemporary designs"},
    {"original": "urban", "equivalent": "city-based structures"},
    {"original": "rural", "equivalent": "countryside areas"},
    {"original": "ferrous", "equivalent": "iron-containing metals"},
    {"original": "valuable", "equivalent": "highly prized assets"},
    {"original": "fragrant", "equivalent": "sweet-smelling substances"},
    {"original": "stagnant", "equivalent": "non-flowing waters"},
    {"original": "durable", "equivalent": "long-lasting products"},
    {"original": "flexible", "equivalent": "highly bendable materials"}
]


def main():
    base_dir = Path(__file__).parent
    complex_dir = base_dir / "Complex"

    input_file = complex_dir / "st3_english_complex.json"

    if not input_file.exists():
        print("File not found:", input_file)
        return

    print(f"Processing file: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)

    for index, item in enumerate(data, 1):
        text = item["syllogism"]
        premises = text.split(".")
        modified_premises = []

        for premise in premises:
            if premise.strip():
                modified_premise = premise

                for mapping in mappings:
                    original = mapping["original"]
                    equivalent = mapping["equivalent"]

                    if original.lower() in modified_premise.lower():
                        if random.random() > 0.5:
                            modified_premise = re.sub(
                                r'\b' + re.escape(original) + r'\b',
                                equivalent,
                                modified_premise,
                                flags=re.IGNORECASE
                            )

                modified_premises.append(modified_premise)

        item["syllogism"] = ".".join(modified_premises)

        print(f"[{index}/{total}] processed", end="\r")

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("\nFile updated successfully.")


if __name__ == "__main__":
    main()
