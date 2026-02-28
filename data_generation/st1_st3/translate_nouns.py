import json
import re
import time
from pathlib import Path
from deep_translator import GoogleTranslator

# CONFIG
LANGUAGES = ['german', 'spanish', 'french', 'italian', 'dutch',
             'portuguese', 'russian', 'chinese', 'swahili', 'bengali', 'telugu']
CHAR_LIMIT = 4000  # Character limit for batch translation
SEPARATOR = "\n"

# Map languages to Google Translator codes
LANGUAGE_CODES = {
    "german": "de",
    "spanish": "es",
    "french": "fr",
    "italian": "it",
    "dutch": "nl",
    "portuguese": "pt",
    "russian": "ru",
    "chinese": "zh-CN",
    "swahili": "sw",
    "bengali": "bn",
    "telugu": "te"
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def precompute_noun_map(nouns, language, cache_path):
    # Load cache if exists
    if cache_path.exists():
        print(f"Loading cached translations from {cache_path.name}")
        noun_map = load_json(cache_path)
    else:
        noun_map = {}

    translator = GoogleTranslator(source="en", target=LANGUAGE_CODES[language])
    remaining_nouns = [n for n in nouns if n not in noun_map]

    if not remaining_nouns:
        return noun_map

    print(f"Translating {len(remaining_nouns)} new nouns for {language}...")
    batch, batch_len = [], 0

    def flush(current_batch):
        joined = SEPARATOR.join(current_batch)
        try:
            translated = translator.translate(joined)
            parts = [p.strip() for p in translated.split(SEPARATOR) if p.strip()]

            if len(parts) != len(current_batch):
                raise ValueError("Mismatch in batch translation")

            for src, tgt in zip(current_batch, parts):
                noun_map[src] = tgt
        except:
            print(f"Batch failed, retrying individual words...")
            for word in current_batch:
                try:
                    noun_map[word] = translator.translate(word)
                    time.sleep(0.3)
                except:
                    noun_map[word] = word  # fallback

        save_json(noun_map, cache_path)
        print(f"Saved progress: {len(noun_map)}/{len(nouns)} nouns.")

    for noun in remaining_nouns:
        size = len(noun.encode("utf-8")) + 1
        if batch_len + size > CHAR_LIMIT:
            flush(batch)
            batch, batch_len = [], 0
            time.sleep(1)
        batch.append(noun)
        batch_len += size

    if batch:
        flush(batch)
    return noun_map

def replace_nouns_fast(text, noun_map):
    words = re.split(r'(\W+)', text)
    return "".join([noun_map.get(w, w) for w in words])

def main():
    base_dir = Path(__file__).parent
    input_file = base_dir / "simple_4_var" / "english_4_var_simple.json"
    nouns_file = base_dir / "simple_4_var" / "shared_unique_nouns.json"
    output_dir = base_dir / "nouns_translated_4_var"
    cache_dir = base_dir / "cache"

    output_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    if not input_file.exists():
        print(f"Could not find {input_file}")
        return
    if not nouns_file.exists():
        print(f"Could not find {nouns_file}")
        return

    nouns = load_json(nouns_file)
    data = load_json(input_file)

    for lang in LANGUAGES:
        print(f"\nTranslating nouns to {lang}...")
        cache_path = cache_dir / f"{lang}_noun_cache.json"
        noun_map = precompute_noun_map(nouns, lang, cache_path)

        translated_data = []
        for i, item in enumerate(data):
            translated_item = item.copy()
            if "syllogism" in item:
                translated_item["syllogism"] = replace_nouns_fast(item["syllogism"], noun_map)
            translated_data.append(translated_item)

            if i % 100 == 0:
                print(f"Processed {i}/{len(data)} items for {lang}...", end="\r")

        output_file = output_dir / f"{input_file.stem}_{lang}.json"
        save_json(translated_data, output_file)
        print(f"Saved translated file: {output_file}")

    print("\nAll translations completed!")

if __name__ == "__main__":
    main()
