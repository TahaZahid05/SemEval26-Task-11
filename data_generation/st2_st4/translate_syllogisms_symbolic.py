import json
import re
from pathlib import Path

# ==============================
# SENTENCE TEMPLATES (UNCHANGED)
# ==============================
SENTENCE_TEMPLATES = {
    "english": {
        "All {A} are {B}.": "All {A} are {B}.",
        "Some {A} are {B}.": "Some {A} are {B}.",
        "Some {A} are not {B}.": "Some {A} are not {B}.",
        "No {A} are {B}.": "No {A} are {B}.",
        "All {D} that are {A} are {B}.": "All {D} that are {A} are {B}.",
        "Some {D} that are {A} are {B}.": "Some {D} that are {A} are {B}.",
        "Some {D} that are {A} are not {B}.": "Some {D} that are {A} are not {B}.",
        "No {D} that are {A} are {B}.": "No {D} that are {A} are {B}.",
    },
    "german": {
        "All {A} are {B}.": "Alle {A} sind {B}.",
        "Some {A} are {B}.": "Einige {A} sind {B}.",
        "Some {A} are not {B}.": "Einige {A} sind nicht {B}.",
        "No {A} are {B}.": "Keine {A} ist {B}.",
        "All {D} that are {A} are {B}.": "Alle {D}, die {A} sind, sind {B}.",
        "Some {D} that are {A} are {B}.": "Einige {D}, die {A} sind, sind {B}.",
        "Some {D} that are {A} are not {B}.": "Einige {D}, die {A} sind, sind nicht {B}.",
        "No {D} that are {A} are {B}.": "Keine {D}, die {A} ist, ist {B}.",
    },
    "spanish": {
        "All {A} are {B}.": "Todos los {A} son {B}.",
        "Some {A} are {B}.": "Algunos {A} son {B}.",
        "Some {A} are not {B}.": "Algunos {A} no son {B}.",
        "No {A} are {B}.": "Ningún {A} es {B}.",
        "All {D} that are {A} are {B}.": "Todos los {D} que son {A} son {B}.",
        "Some {D} that are {A} are {B}.": "Algunos {D} que son {A} son {B}.",
        "Some {D} that are {A} are not {B}.": "Algunos {D} que son {A} no son {B}.",
        "No {D} that are {A} are {B}.": "Ningún {D} que es {A} es {B}.",
    },
    "french": {
        "All {A} are {B}.": "Tous les {A} sont {B}.",
        "Some {A} are {B}.": "Certains {A} sont {B}.",
        "Some {A} are not {B}.": "Certains {A} ne sont pas {B}.",
        "No {A} are {B}.": "Aucun {A} n'est {B}.",
        "All {D} that are {A} are {B}.": "Tous les {D} qui sont {A} sont {B}.",
        "Some {D} that are {A} are {B}.": "Certains {D} qui sont {A} sont {B}.",
        "Some {D} that are {A} are not {B}.": "Certains {D} qui sont {A} ne sont pas {B}.",
        "No {D} that are {A} are {B}.": "Aucun {D} qui sont {A} n'est {B}.",
    },
    "italian": {
        "All {A} are {B}.": "Tutti gli {A} sono {B}.",
        "Some {A} are {B}.": "Alcuni {A} sono {B}.",
        "Some {A} are not {B}.": "Alcuni {A} non sono {B}.",
        "No {A} are {B}.": "Nessun {A} è {B}.",
        "All {D} that are {A} are {B}.": "Tutti i {D} che sono {A} sono {B}.",
        "Some {D} that are {A} are {B}.": "Alcuni {D} che sono {A} sono {B}.",
        "Some {D} that are {A} are not {B}.": "Alcuni {D} che sono {A} non sono {B}.",
        "No {D} that are {A} are {B}.": "Nessun {D} che è {A} è {B}.",
    },
    "dutch": {
        "All {A} are {B}.": "Alle {A} zijn {B}.",
        "Some {A} are {B}.": "Sommige {A} zijn {B}.",
        "Some {A} are not {B}.": "Sommige {A} zijn niet {B}.",
        "No {A} are {B}.": "Geen {A} zijn {B}.",
        "All {D} that are {A} are {B}.": "Alle {D} die {A} zijn, zijn {B}.",
        "Some {D} that are {A} are {B}.": "Sommige {D} die {A} zijn, zijn {B}.",
        "Some {D} that are {A} are not {B}.": "Sommige {D} die {A} zijn, zijn niet {B}.",
        "No {D} that are {A} are {B}.": "Geen {D} die {A} zijn, zijn {B}.",
    },
    "portuguese": {
        "All {A} are {B}.": "Todos os {A} são {B}.",
        "Some {A} are {B}.": "Alguns {A} são {B}.",
        "Some {A} are not {B}.": "Alguns {A} não são {B}.",
        "No {A} are {B}.": "Nenhum {A} é {B}.",
        "All {D} that are {A} are {B}.": "Todos os {D} que são {A} são {B}.",
        "Some {D} that are {A} are {B}.": "Alguns {D} que são {A} são {B}.",
        "Some {D} that are {A} are not {B}.": "Alguns {D} que são {A} não são {B}.",
        "No {D} that are {A} are {B}.": "Nenhum {D} que é {A} é {B}.",
    },
    "russian": {
        "All {A} are {B}.": "Все {A} являются {B}.",
        "Some {A} are {B}.": "Некоторые {A} являются {B}.",
        "Some {A} are not {B}.": "Некоторые {A} не являются {B}.",
        "No {A} are {B}.": "Нет {A}, которые являются {B}.",
        "All {D} that are {A} are {B}.": "Все {D}, которые являются {A}, являются {B}.",
        "Some {D} that are {A} are {B}.": "Некоторые {D}, которые являются {A}, являются {B}.",
        "Some {D} that are {A} are not {B}.": "Некоторые {D}, которые являются {A}, не являются {B}.",
        "No {D} that are {A} are {B}.": "Нет {D}, которые являются {A}, являются {B}",
    },
    "chinese": {
        "All {A} are {B}.": "所有{A}都是{B}。",
        "Some {A} are {B}.": "有些{A}是{B}。",
        "Some {A} are not {B}.": "有些{A}不是{B}。",
        "No {A} are {B}.": "没有{A}是{B}。",
        "All {D} that are {A} are {B}.": "所有{D}都是{A}，它们都是{B}。",
        "Some {D} that are {A} are {B}.": "有些{D}是{A}，它们是{B}。",
        "Some {D} that are {A} are not {B}.": "有些{D}是{A}，它们不是{B}。",
        "No {D} that are {A} are {B}.": "没有{D}是{A}，它们是{B}。", 
    },
    "swahili": {
        "All {A} are {B}.": "Zote {A} ni {B}.",
        "Some {A} are {B}.": "Baadhi ya {A} ni {B}.",
        "Some {A} are not {B}.": "Baadhi ya {A} si {B}.",
        "No {A} are {B}.": "Hakuna {A} ni {B}.",
        "All {D} that are {A} are {B}.": "{D} zote ambazo ni {A} ni {B}.",
        "Some {D} that are {A} are {B}.": "Baadhi ya {D} ambazo ni {A} ni {B}.",
        "Some {D} that are {A} are not {B}.": "Baadhi ya {D} ambazo ni {A} si {B}.",
        "No {D} that are {A} are {B}.": "Hakuna {D} ambazo ni {A} ni {B}.",
    },
    "bengali": {
        "All {A} are {B}.": "সব {A} হল {B}।",
        "Some {A} are {B}.": "কিছু {A} হল {B}।",
        "Some {A} are not {B}.": "কিছু {A} হল {B}।",
        "No {A} are {B}.": "না {A} হল {B}।",
        "All {D} that are {A} are {B}.": "সব {D} হল {A} হল {B}।",
        "Some {D} that are {A} are {B}.": "কিছু {D} হল {A} হল {B}।",
        "Some {D} that are {A} are not {B}.": "কিছু {D} হল {A} হল {B}।",
        "No {D} that are {A} are {B}.": "না {D} হল {A} হল {B}।",
    },
    "telugu": {
        "All {A} are {B}.": "అన్ని {A} లు {B}.",
        "Some {A} are {B}.": "కొన్ని {A} లు {B}.",
        "Some {A} are not {B}.": "కొన్ని {A} లు {B} కావు.",
        "No {A} are {B}.": "ఏ {A} లు {B} కావు.",
        "All {D} that are {A} are {B}.": "{A} లు ఉన్న అన్ని {D} లు {B} అవుతాయి.",
        "Some {D} that are {A} are {B}.": "{A} లు ఉన్న కొన్ని {D} లు {B} అవుతాయి.",
        "Some {D} that are {A} are not {B}.": "{A} లు ఉన్న కొన్ని {D} లు {B} కావు.",
        "No {D} that are {A} are {B}.": "{A} లు ఉన్న ఏ {D} లు {B} అవుతాయి.",
    },
}
# (keep the rest of SENTENCE_TEMPLATES exactly as you wrote them)

# ==============================
# REGEX PATTERNS (UNCHANGED)
# ==============================
SENTENCE_PATTERNS = [
    # 4-variable patterns (check these first)
    (r"All (.+?) that are (.+?) are (.+?)\.", "All {D} that are {A} are {B}."),
    (r"Some (.+?) that are (.+?) are not (.+?)\.", "Some {D} that are {A} are not {B}."),  # Check "are not" BEFORE "are"
    (r"Some (.+?) that are (.+?) are (.+?)\.", "Some {D} that are {A} are {B}."),
    (r"No (.+?) that are (.+?) are (.+?)\.", "No {D} that are {A} are {B}."),
    # 3-variable patterns - MUST check "are not" patterns BEFORE "are" patterns
    (r"All (.+?) are (.+?)\.", "All {A} are {B}."),
    (r"Some (.+?) are not (.+?)\.", "Some {A} are not {B}."),  # Check "are not" BEFORE "are"
    (r"Some (.+?) are (.+?)\.", "Some {A} are {B}."),
    (r"No (.+?) are (.+?)\.", "No {A} are {B}."),
]


def identify_sentence_type(sentence):
    sentence = sentence.strip()
    for pattern, template_key in SENTENCE_PATTERNS:
        match = re.match(pattern, sentence)
        if match:
            groups = match.groups()
            if "{D}" in template_key:
                return template_key, {"D": groups[0], "A": groups[1], "B": groups[2]}
            else:
                return template_key, {"A": groups[0], "B": groups[1]}
    return None, None


def translate_sentence(sentence, language):
    template_key, nouns = identify_sentence_type(sentence)
    if template_key is None:
        print(f"Unmatched sentence: {sentence}")
        return sentence

    template = SENTENCE_TEMPLATES[language][template_key]
    for k, v in nouns.items():
        template = template.replace(f"{{{k}}}", v)
    return template


def translate_syllogism(syllogism, language):
    sentences = [s.strip() + "." for s in syllogism.split(". ") if s.strip()]
    return " ".join(translate_sentence(s, language) for s in sentences)


# ==============================
# MAIN (MATCHES NON-SYMBOLIC LOGIC)
# ==============================
def main():
    merged_dir = Path("ST4_Simple_Symbolic_All")

    if not merged_dir.exists():
        raise FileNotFoundError("SimpleMerged directory not found")

    # Only pick symbolic files
    json_files = sorted(merged_dir.glob("*_symbolic.json"))

    print(f"Found {len(json_files)} symbolic files.")

    for file_path in json_files:
        # Language is always the second part: st4_<language>_simple_symbolic.json
        language = file_path.stem.split("_")[0].lower()

        if language not in SENTENCE_TEMPLATES:
            print(f"Skipping {file_path.name} (unknown language)")
            continue

        print(f"\nProcessing {file_path.name} → {language}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if "syllogism" in item:
                item["syllogism"] = translate_syllogism(
                    item["syllogism"],
                    language
                )

        output_file = file_path.with_name(
            file_path.stem + "_translated.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"Saved {output_file.name}")

    print("\nSymbolic translation complete!")




if __name__ == "__main__":
    main()
