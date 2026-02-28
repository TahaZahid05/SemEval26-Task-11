import json
import re
import sys
import time
from pathlib import Path
from deep_translator import GoogleTranslator

# Language code mapping for deep-translator
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
    "telugu": "te",
}

# Cache for translated nouns to avoid repeated API calls
translation_cache = {}
precomputed = False  # Flag to track if precomputation was done

# Define sentence templates for all languages
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
    # Alle {A} sind {B}. Einige {A} sind {B}. Einige {A} sind nicht {B}. Kein {A} ist {B}. Alle {D}, die {A} sind, sind {B}. Einige {D}, die {A} sind, sind {B}. Einige {D}, die {A} sind, sind nicht {B}. Kein {D}, das {A} ist, ist {B}.
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
    # Todos los {A} son {B}. Algunos {A} son {B}. Algunos {A} no son {B}. Ningún {A} es {B}. Todos los {D} que son {A} son {B}. Algunos {D} que son {A} son {B}. Algunos {D} que son {A} no son {B}. Ningún {D} que sea {A} es {B}.
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
    # Tous les {A} sont {B}. Certains {A} sont {B}. Certains {A} ne sont pas {B}. Aucun {A} n'est {B}. Tous les {D} qui sont {A} sont {B}. Certains {D} qui sont {A} sont {B}. Certains {D} qui sont {A} ne sont pas {B}. Aucun {D} qui est {A} n'est {B}.
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
    # Tutti gli {A} sono {B}. Alcuni {A} sono {B}. Alcuni {A} non sono {B}. Nessun {A} è {B}. Tutti i {D} che sono {A} sono {B}. Alcuni {D} che sono {A} sono {B}. Alcuni {D} che sono {A} non sono {B}. Nessun {D} che è {A} è {B}.
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
    # Alle {A} zijn {B}. Sommige {A} zijn {B}. Sommige {A} zijn niet {B}. Geen {A} zijn {B}. Alle {D} die {A} zijn, zijn {B}. Sommige {D} die {A} zijn, zijn {B}. Sommige {D} die {A} zijn, zijn niet {B}. Geen {D} die {A} zijn, zijn {B}.
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
    # Todos os {A} são {B}. Alguns {A} são {B}. Alguns {A} não são {B}. Nenhum {A} é {B}. Todos os {D} que são {A} são {B}. Alguns {D} que são {A} são {B}. Alguns {D} que são {A} não são {B}. Nenhum {D} que é {A} é {B}.
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
    # Все {A} являются {B}. Некоторые {A} являются {B}. Некоторые {A} не являются {B}. Ни один {A} не является {B}. Все {D}, которые являются {A}, являются {B}. Некоторые {D}, которые являются {A}, являются {B}. Некоторые {D}, которые являются {A}, не являются {B}. Ни один {D}, которые являются {A}, не является {B}.
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
    # 所有{A}都是{B}。有些{A}是{B}。有些{A}不是{B}。没有{A}是{B}。所有{D}都是{A}，它们都是{B}。有些{D}是{A}，它们是{B}。有些{D}是{A}，它们不是{B}。没有{D}是{A}，它们是{B}。
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
    # Zote {A} ni {B}. Baadhi ya {A} ni {B}. Baadhi ya {A} si {B}. Hakuna {A} ni {B}. {D} zote ambazo ni {A} ni {B}. Baadhi ya {D} ambazo ni {A} ni {B}. Baadhi ya {D} ambazo ni {A} si {B}. Hakuna {D} ambazo ni {A} ni {B}.
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
    # সব {A} হল {B}। কিছু {A} হল {B}। কিছু {A} হল {B}। না {A} হল {B}। সব {D} হল {A} হল {B}। কিছু {D} হল {A} হল {B}। কিছু {D} হল {A} হল {B}। কিছু {D} হল {A} হল {B}। না {D} হল {A} হল {B}।
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
    # అన్ని {A} లు {B}. కొన్ని {A} లు {B}. కొన్ని {A} లు {B} కావు. ఏ {A} లు {B} కావు. ఏ {A} లు {B} కావు. {A} లు ఉన్న అన్ని {D} లు {B} అవుతాయి. {A} లు ఉన్న కొన్ని {D} లు {B} అవుతాయి. {A} లు ఉన్న కొన్ని {D} లు {B} కావు. {A} లు ఉన్న ఏ {D} లు {B} అవుతాయి.
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

# Define regex patterns to match sentence types
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

def detect_language_from_filename(filename):
    filename = filename.lower()
    for lang in LANGUAGE_CODES.keys():
        if lang in filename:
            return lang
    return None


def translate_noun(noun, target_language):
    """
    Translate a single noun using cached translations or deep-translator.
    If precomputed, only use cache and don't make new API calls.
    """
    # Check cache first
    cache_key = f"{noun}_{target_language}"
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    
    # If translating to English, return as-is
    if target_language == "english":
        translation_cache[cache_key] = noun
        return noun
    
    # If we precomputed, only use cached values
    if precomputed:
        # If still not found, use original noun
        translation_cache[cache_key] = noun
        return noun
    
    # Otherwise, translate on-the-fly (when not using precomputation)
    target_code = LANGUAGE_CODES[target_language]
    
    try:
        translator = GoogleTranslator(source='en', target=target_code)
        translated = translator.translate(noun)
        
        # Cache the result
        translation_cache[cache_key] = translated
        return translated
    except Exception as e:
        print(f"Warning: Translation failed for '{noun}' to {target_language}: {e}")
        translation_cache[cache_key] = noun
        return noun


def identify_sentence_type(sentence):
    """
    Identify the sentence type and extract nouns.
    Returns: (template_key, nouns_dict) or (None, None) if not matched
    """
    sentence = sentence.strip()
    
    for pattern, template_key in SENTENCE_PATTERNS:
        match = re.match(pattern, sentence)
        if match:
            groups = match.groups()
            if template_key.startswith("All {D}") or template_key.startswith("Some {D}") or template_key.startswith("No {D}"):
                # 4-variable pattern: D, A, B
                return template_key, {"D": groups[0], "A": groups[1], "B": groups[2]}
            else:
                # 3-variable pattern: A, B
                return template_key, {"A": groups[0], "B": groups[1]}
    
    return None, None


def translate_sentence(sentence, language):
    """
    Translate a single sentence to the target language.
    """
    # Identify sentence type and extract nouns
    template_key, nouns = identify_sentence_type(sentence)
    
    if template_key is None:
        print(f"Warning: Could not identify sentence type for: {sentence}")
        return sentence
    
    # Get the translated template
    translated_template = SENTENCE_TEMPLATES[language][template_key]
    
    # Translate each noun
    translated_nouns = {}
    for placeholder, noun in nouns.items():
        translated_nouns[placeholder] = translate_noun(noun, language)
    
    # Replace placeholders in template
    translated_sentence = translated_template
    for placeholder, translated_noun in translated_nouns.items():
        translated_sentence = translated_sentence.replace(f"{{{placeholder}}}", translated_noun)
    
    return translated_sentence


def translate_syllogism(syllogism, language):
    """
    Translate an entire syllogism to the target language.
    """
    # Split by period followed by space (sentence boundary)
    sentences = [s.strip() + "." for s in syllogism.split(". ") if s.strip()]
    
    # Translate each sentence
    translated_sentences = []
    for sentence in sentences:
        translated = translate_sentence(sentence, language)
        translated_sentences.append(translated)
    
    # Join sentences with space
    return " ".join(translated_sentences)


def precompute_translations(nouns_file, languages):
    """
    Precompute translations for all nouns using batch blob technique with character limit handling.
    Splits nouns into batches respecting Google Translate's 5000 character limit.
    """
    global precomputed
    
    print(f"\nLoading unique nouns from {nouns_file}...")
    with open(nouns_file, 'r', encoding='utf-8') as f:
        unique_nouns = json.load(f)
    
    print(f"Found {len(unique_nouns)} unique nouns to translate.")
    print("Precomputing translations for all languages (using batch blob technique)...\n")
    
    # Separator that won't appear in noun names
    separator = " ||| "
    char_limit = 5000  # Google Translate character limit
    
    # Pre-compute batches for all languages
    batches = []
    current_batch = []
    current_length = 0
    
    for noun in unique_nouns:
        noun_with_sep = noun + separator
        noun_length = len(noun_with_sep.encode('utf-8'))
        
        # If adding this noun would exceed limit, start new batch
        if current_length + noun_length > char_limit and current_batch:
            batches.append(current_batch)
            current_batch = [noun]
            current_length = noun_length
        else:
            current_batch.append(noun)
            current_length += noun_length
    
    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)
    
    print(f"Split {len(unique_nouns)} nouns into {len(batches)} batches (max 5000 chars each)\n")
    
    for language in languages:
        if language == "english":
            # English nouns map to themselves
            for noun in unique_nouns:
                cache_key = f"{noun}_english"
                translation_cache[cache_key] = noun
            print(f"Cached {len(unique_nouns)} English nouns (identity mapping)")
            continue
        
        try:
            target_code = LANGUAGE_CODES[language]
            print(f"  Translating to {language}...", end=" ", flush=True)
            
            translator = GoogleTranslator(source='en', target=target_code)
            translated_count = 0
            batch_success = 0
            
            # Process each batch
            for batch_idx, batch in enumerate(batches):
                # Join batch into blob
                joined_text = separator.join(batch)
                
                try:
                    # Translate the blob
                    translated_blob = translator.translate(joined_text)
                    
                    # Split back into individual translations
                    translated_batch = translated_blob.split(separator)
                    
                    # Handle case where separator might not be perfectly preserved
                    if len(translated_batch) != len(batch):
                        # Fallback: use newline as separator
                        joined_text = "\n".join(batch)
                        translated_blob = translator.translate(joined_text)
                        translated_batch = translated_blob.split("\n")
                    
                    # Verify we got the right number of translations
                    if len(translated_batch) != len(batch):
                        raise ValueError(f"Batch split mismatch: expected {len(batch)}, got {len(translated_batch)}")
                    
                    # Cache all translations from this batch
                    for original, translated in zip(batch, translated_batch):
                        cache_key = f"{original}_{language}"
                        translation_cache[cache_key] = translated.strip()
                        translated_count += 1
                    
                    batch_success += 1
                
                except Exception as batch_error:
                    print(f"\n    Error in batch {batch_idx + 1}/{len(batches)}: {batch_error}")
                    # Try to cache originals as fallback
                    for original in batch:
                        cache_key = f"{original}_{language}"
                        translation_cache[cache_key] = original
                    translated_count += len(batch)
            
            print(f"[OK] Cached {translated_count} translations in {batch_success}/{len(batches)} successful batch(es)")
            
            # Rate limiting - be respectful to Google's API
            time.sleep(1)
            
        except Exception as e:
            print(f"\n    Error translating to {language}: {e}")
            print(f"    Skipping {language}...")
    
    # Mark precomputation as complete
    precomputed = True
    print(f"\nPrecomputation complete! Total cached: {len(translation_cache)} translations.")


def main():

    merged_dir = Path("ST4_Simple_All")  # <-- Set your input folder here

    if not merged_dir.is_dir():
        print("Please provide a valid SimpleMerged directory.")
        sys.exit(1)

    # Create output folder as a sibling of the input folder
    output_dir = merged_dir.parent / (merged_dir.name + "_translated")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_dir}")

    # Locate shared_unique_nouns.json in the current working directory
    nouns_file = Path.cwd() / "shared_unique_nouns.json"

    if nouns_file.exists():
        print(f"Found shared nouns file: {nouns_file}")
        languages = list(LANGUAGE_CODES.keys())
        precompute_translations(nouns_file, languages)
    else:
        print("shared_unique_nouns.json not found in current directory — running without precomputation.")

    input_files = sorted(merged_dir.glob("*.json"))

    if not input_files:
        print("No JSON files found in input folder.")
        sys.exit(1)

    print(f"\nFound {len(input_files)} files to translate.\n")

    for input_file in input_files:
        language = detect_language_from_filename(input_file.name)

        if language is None:
            print(f"Could not detect language from filename: {input_file.name}")
            continue

        print(f"Processing {input_file.name} → {language}")

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        translated_data = []
        for item in data:
            translated_item = item.copy()

            if "syllogism" in item:
                translated_item["syllogism"] = translate_syllogism(
                    item["syllogism"],
                    language
                )

            translated_data.append(translated_item)

        output_path = output_dir / (input_file.stem + "_translated.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, indent=4, ensure_ascii=False)

        print(f"Saved → {output_path.name}")

if __name__ == "__main__":
    main()