import json
import re
import time
from pathlib import Path
from deep_translator import GoogleTranslator


# CONFIG - Add as many as you need here
LANGUAGE_CODES = {
    "english": "en",
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

CHAR_LIMIT = 4000 
SEPARATOR = "\n"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def precompute_noun_map(nouns, language_name, lang_code, cache_dir):
    cache_path = cache_dir / f"{language_name}_noun_cache.json"
    
    if cache_path.exists():
        print(f"Loading cached {language_name} translations...")
        noun_map = load_json(cache_path)
    else:
        noun_map = {}

    translator = GoogleTranslator(source="en", target=lang_code)
    remaining_nouns = [n for n in nouns if n not in noun_map]
    
    if not remaining_nouns:
        return noun_map

    print(f"Translating {len(remaining_nouns)} new nouns for {language_name}...")
    batch, batch_len = [], 0
    
    def flush(current_batch):
        joined = SEPARATOR.join(current_batch)
        try:
            translated = translator.translate(joined)
            parts = [p.strip() for p in translated.split(SEPARATOR) if p.strip()]
            
            if len(parts) != len(current_batch):
                raise ValueError("Mismatch")
                
            for src, tgt in zip(current_batch, parts):
                noun_map[src] = tgt
        except Exception as e:
            print(f"Batch failed for {language_name}, retrying individual words...")
            for word in current_batch:
                try:
                    noun_map[word] = translator.translate(word)
                    time.sleep(0.3)
                except: 
                    noun_map[word] = word 

        save_json(noun_map, cache_path)

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
    if not noun_map:
        return text

    # 1. Sort keys by length (descending) 
    # This is CRITICAL: We must match "physical entities" before "physical"
    sorted_keys = sorted(noun_map.keys(), key=len, reverse=True)
    
    # 2. Create a regex pattern that matches any of the nouns
    # \b ensures we match whole words only, re.escape handles special characters
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_keys)) + r')\b')

    # 3. Use a callback function to do the replacement
    return pattern.sub(lambda x: noun_map[x.group()], text)

def main():
    base_dir = Path(__file__).parent
    input_dir = base_dir / "ST4_Simple_One"
    nouns_file = base_dir / "shared_unique_nouns.json"
    output_base_dir = base_dir / "ST4_Simple_One_Translated_Nouns"
    cache_dir = base_dir / "cache"
    
    output_base_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    if not nouns_file.exists():
        print(f"Could not find {nouns_file}")
        return

    shared_nouns = load_json(nouns_file)

    for lang_name, lang_code in LANGUAGE_CODES.items():
        print(f"\n--- Processing Language: {lang_name.upper()} ---")
        
        # 1. Get/Update translations for this specific language
        noun_map = precompute_noun_map(shared_nouns, lang_name, lang_code, cache_dir)

        # 2. Find all files for this language (e.g., files with 'telugu' in the name)
        # Adjust the glob pattern if your naming convention is different
        files_to_process = list(input_dir.glob(f"*{lang_name}*.json"))

        if not files_to_process:
            print(f"No files found for {lang_name} in {input_dir}")
            continue

        for input_file in files_to_process:
            print(f"Applying translations to {input_file.name}...")
            data = load_json(input_file)
            
            for i, item in enumerate(data):
                if "syllogism" in item:
                    item["syllogism"] = replace_nouns_fast(item["syllogism"], noun_map)
                if i % 200 == 0:
                    print(f"{lang_name}: Processed {i}/{len(data)} items...", end="\r")

            save_json(data, output_base_dir / input_file.name)
            print(f"\nFinished {input_file.name}")

    print("\nAll languages and files processed!")

if __name__ == "__main__":
    main()