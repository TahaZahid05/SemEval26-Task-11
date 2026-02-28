import random
import time
import uuid
import json
import nltk
import re
from nltk.corpus import wordnet as wn
try:
    wn.synsets('dog')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

from collections import defaultdict
import pandas as pd
import inflect
p = inflect.engine()

FREQUENCY_THRESHOLD = 5
count = 0

print("Pre-loading and Filtering Common WordNet Nouns...")
start_time = time.time() # Assuming 'import time' is present

# 1. Initialize the list to store the filtered synsets
common_synsets = []

# 2. Iterate over all noun synsets
for synset in wn.all_synsets('n'):
    # Check 1: Filter out the absolute top-level 'entity' synset
    if not synset.hypernyms():
        continue 
    
    # Check 2: Get the primary lemma and check its corpus frequency count
    # We use the first lemma as the representative term for frequency
    if synset.lemmas():
        lemma = synset.lemmas()[0]
        
        # Keep the synset only if the lemma count meets the threshold
        if lemma.count() >= FREQUENCY_THRESHOLD:
            common_synsets.append(synset)

# --- Output Variables ---

# GLOBAL CACHE: Load this once here, never inside a function
ALL_NOUN_SYNSETS = common_synsets
print(f"Loaded and Filtered **{len(ALL_NOUN_SYNSETS):,}** common synsets (Threshold: {FREQUENCY_THRESHOLD}).")

# Helper to speed up exclusion checks (sets are faster than lists)
ALL_NOUN_NAMES = {s.name().split('.')[0] for s in ALL_NOUN_SYNSETS}

# --- Utility functions ---

def wn_name(syn):
    """Clean up WordNet synset name."""
    return syn.lemmas()[0].name().replace("_", " ")

def get_unrelated_noun(exclude_synsets, max_attempts=500):
    """
    Pick a random noun unrelated to the given synsets using the global cache.
    """
    # OPTIMIZATION: Use the global set of names for O(1) lookup
    exclude_names = {s.name().split('.')[0] for s in exclude_synsets}

    for attempt in range(1, max_attempts + 1):
        # OPTIMIZATION: Use the pre-loaded global list
        candidate = random.choice(ALL_NOUN_SYNSETS)
        cand_name = candidate.name().split('.')[0]

        # Fast string check
        if cand_name in exclude_names:
            continue

        # Compute closure with limit
        # (This logic remains the same, but runs faster because we reach here faster)
        related = set()
        queue = [candidate]
        depth = 0
        
        found_conflict = False
        while queue and depth < 3:  # limit traversal depth
            next_level = []
            for node in queue:
                # Check current node immediately
                if node.name().split('.')[0] in exclude_names:
                     found_conflict = True
                     break
                     
                for rel in node.hypernyms() + node.hyponyms():
                    if rel not in related:
                        related.add(rel)
                        next_level.append(rel)
            if found_conflict: break
            queue = next_level
            depth += 1
        
        if found_conflict:
            continue

        # Final check on the expanded set
        # OPTIMIZATION: Check against the set of names, not synset objects
        related_names = {r.name().split('.')[0] for r in related}
        if not related_names.intersection(exclude_names):
            return candidate   


    raise RuntimeError(f"Failed to find unrelated noun after {max_attempts} attempts.")

def get_hypernym_chain(synset, depth):
    """Return a chain like A -> B using hypernyms from WordNet (depth=1 now)."""
    chain = [synset]
    current = synset
    for _ in range(depth):
        hypers = current.hypernyms()
        if not hypers:
            break
        current = random.choice(hypers)
        chain.append(current)
    return chain

def generate_syllogisms(schema, chain_type, validity, plausibility, count=1, used_nouns=None):
    """
    Generate syllogisms using given templates and parameters.
    Supports both plausible and not plausible schema types.
    Returns a list of JSON-like dicts.
    """
    results = []

    curr_ind = 0
    inner_ind = 0
    all_labels = list(schema.keys())

    for _ in range(count):
        while True:
            base_synset = random.choice(ALL_NOUN_SYNSETS)
            related_chain_clean = []
            curr_used = []

            # --- SAME CHAIN (Valid or Invalid, Plausible or Not Plausible) ---
            if chain_type in [
                "valid_p_same_chain",
                "invalid_p_same_chain",
                "valid_np_same_chain",
                "invalid_np_same_chain"
            ]:
                chain = get_hypernym_chain(base_synset, depth=2)
                if len(chain) < 3:
                    continue
                A, B, C = map(wn_name, chain)
                curr_used = chain
                related_chain_clean.append(A)
                related_chain_clean.append(B)
                related_chain_clean.append(C)
                # D is the superset (hypernym) of all words in the chain
                if chain[-1].hypernyms():
                    D = wn_name(chain[-1].hypernyms()[0])
                else:
                    continue

            # --- DIFFERENT CHAIN (C differs) ---
            elif chain_type in [
                "valid_p_diff_chain_C",
                "invalid_p_diff_chain_C",
                "valid_np_diff_chain_C",
                "invalid_np_diff_chain_C"
            ]:
                chain = get_hypernym_chain(base_synset, depth=1)
                if len(chain) < 2:
                    continue
                unrelated_syn = get_unrelated_noun(chain)
                curr_used = chain + [unrelated_syn]
                A, B, C = map(wn_name, chain + [unrelated_syn])
                related_chain_clean.append(A)
                related_chain_clean.append(B)
                # D is the superset (hypernym) of all words in the chain
                if chain[-1].hypernyms():
                    D = wn_name(chain[-1].hypernyms()[0])
                else:
                    continue

            # --- DIFFERENT CHAIN (B differs) ---
            elif chain_type in [
                "invalid_p_diff_chain_B",
                "valid_np_diff_chain_B",
                "invalid_np_diff_chain_B"
            ]:
                chain = get_hypernym_chain(base_synset, depth=1)
                if len(chain) < 2:
                    continue
                unrelated_syn = get_unrelated_noun(chain)
                A = wn_name(chain[0])
                B = wn_name(unrelated_syn)
                C = wn_name(chain[1])
                curr_used = chain + [unrelated_syn]
                # D is the superset (hypernym) of all words in the chain
                related_chain_clean.append(A)
                related_chain_clean.append(C)
                if chain[-1].hypernyms():
                    D = wn_name(chain[-1].hypernyms()[0])
                else:
                    continue

            # --- DIFFERENT CHAIN (A differs) ---
            elif chain_type in [
                "invalid_p_diff_chain_A",
                "valid_np_diff_chain_A",
                "invalid_np_diff_chain_A",
                "valid_p_diff_chain_A"
            ]:
                chain = get_hypernym_chain(base_synset, depth=1)
                if len(chain) < 2:
                    continue
                unrelated_syn = get_unrelated_noun(chain)
                A = wn_name(unrelated_syn)
                B = wn_name(chain[0])
                C = wn_name(chain[1])
                curr_used = chain + [unrelated_syn]
                # D is the superset (hypernym) of all words in the chain
                related_chain_clean.append(B)
                related_chain_clean.append(C)
                if chain[-1].hypernyms():
                    D = wn_name(chain[-1].hypernyms()[0])
                else:
                    continue

            # -- All different chains
            elif chain_type in [
                "valid_np_diff_chain",
                "invalid_p_diff_chain",
                "invalid_np_diff_chain"
            ]:
                chain = [base_synset]
                temp_chain = [base_synset]
                unrelated_syn_1 = get_unrelated_noun(temp_chain)
                temp_chain = temp_chain + [unrelated_syn_1]
                unrelated_syn_2 = get_unrelated_noun(temp_chain)
                temp_chain = temp_chain + [unrelated_syn_2]
                curr_used = temp_chain
                A, B, C = map(wn_name, temp_chain)
                # D is the superset (hypernym) - for all different chains, use base synset's hypernym
                related_chain_clean.append(A)
                if base_synset.hypernyms():
                    D = wn_name(base_synset.hypernyms()[0])
                else:
                    continue
                

            else:
                raise ValueError(f"Unknown chain_type: {chain_type}")

            prem1, prem2, conclusions = schema[all_labels[curr_ind]]

            values_for_first = re.findall(r"\{(.*?)\}", prem1)
            values_for_second = re.findall(r"\{(.*?)\}", prem2)

            first_first, first_second = values_for_first
            second_first, second_second = values_for_second
            
            bool_a = False
            bool_b = False
            bool_c = False

            if not plausibility:
                D = wn_name(get_unrelated_noun(curr_used))
                bool_a, bool_b, bool_c = True, True, True
            
            
            
            if plausibility and A in related_chain_clean:
                bool_a = True

            if plausibility and B in related_chain_clean:
                bool_b = True

            if plausibility and C in related_chain_clean:
                bool_c = True

        

            A, B, C, D = p.plural(A), p.plural(B), p.plural(C), p.plural(D)
            if used_nouns is not None:
                used_nouns.update([A.lower(), B.lower(), C.lower(), D.lower()])

            A_four, B_four_first = A, B
            B_four_second, C_four = B, C 
            # if bool_a:
            #         A_four = f"{D} that are {A}"
            
            # if bool_b:
            #         B_four = f"{D} that are {B}"
            
            # if bool_c:
            #         C_four = f"{D} that are {C}"

            if first_first == "A" and bool_a:
                A_four = f"{D} that are {A}"
            elif first_first == "B" and bool_b:
                B_four_first = f"{D} that are {B}"
            
            if second_first == "B" and bool_b:
                B_four_second = f"{D} that are {B}"
            elif second_first == "C" and bool_c:
                C_four = f"{D} that are {C}"
            
            
            syllogism = f"{prem1.format(A=A_four, B=B_four_first, C=C)} {prem2.format(A=A_four, B=B_four_second, C=C_four)} {conclusions[inner_ind].format(A=A, B=B, C=C)}"

            results.append({
                "id": str(uuid.uuid4()),
                "syllogism": syllogism,
                "validity": validity,
                "plausibility": plausibility
            })

            # Cycle through templates
            if inner_ind == len(conclusions) - 1:
                curr_ind += 1
                inner_ind = 0
            else:
                inner_ind += 1

            if curr_ind == len(all_labels):
                curr_ind = 0
                inner_ind = 0
            break         

    return results


# Naming convention
# x_y_z
# x = valid/invalid
# y = plausible/implausible
# z = different chain / different chain X / same chain



valid_plausible_diff_chain_C = {
    "AE1": ("All {A} are {B}.", "No {B} are {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AE3": ("All {A} are {B}.", "No {C} are {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AO3": ("All {A} are {B}.", "Some {C} are not {B}.", ["Some {C} are not {A}."]),
    "IE1": ("Some {A} are {B}.", "No {B} are {C}.", ["Some {A} are not {C}."]),
    "IE2": ("Some {B} are {A}.", "No {C} are {B}.", ["Some {A} are not {C}."]),
    "IE3": ("Some {A} are {B}.", "No {C} are {B}.", ["Some {A} are not {C}."]),
    "IE4": ("Some {B} are {A}.", "No {B} are {C}.", ["Some {A} are not {C}."])
}

valid_plausible_diff_chain_A ={
    "EA1": ("No {A} are {B}.", "All {B} are {C}.", ["Some {C} are not {A}."]),
    "EA4": ("No {B} are {A}.", "All {B} are {C}.", ["Some {C} are not {A}."]),
    "EI1": ("No {A} are {B}.", "Some {B} are {C}.", ["Some {C} are not {A}."]),
    "EI2": ("No {B} are {A}.", "Some {C} are {B}.", ["Some {C} are not {A}."]),
    "EI3": ("No {A} are {B}.", "Some {C} are {B}.", ["Some {C} are not {A}."]),
    "EI4": ("No {B} are {A}.", "Some {B} are {C}.", ["Some {C} are not {A}."]),
    "OA4": ("Some {B} are not {A}.", "All {B} are {C}.", ["Some {C} are not {A}."])
}

valid_plausible_same_chain = {
    "AA1": ("All {A} are {B}.", "All {B} are {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AO3": ("All {A} are {B}.", "Some {C} are not {B}.", ["Some {C} are not {A}."]),
    "IA1": ("Some {A} are {B}.", "All {B} are {C}.", ["Some {A} are {C}.", "Some {C} are {A}."]),
    "IA4": ("Some {B} are {A}.", "All {B} are {C}.", ["Some {A} are {C}.", "Some {C} are {A}."]),
    "OA4": ("Some {B} are not {A}.", "All {B} are {C}.", ["Some {C} are not {A}."]),
}

invalid_plausible_same_chain = {
    "AI1": ("All {A} are {B}.", "Some {B} are {C}.", ["All {A} are {C}." , "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "AI3": ("All {A} are {B}.", "Some {C} are {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "II1": ("Some {A} are {B}.", "Some {B} are {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "II2": ("Some {B} are {A}.", "Some {C} are {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "II3": ("Some {A} are {B}.", "Some {C} are {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "II4": ("Some {B} are {A}.", "Some {B} are {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IO2": ("Some {B} are {A}.", "Some {C} are not {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IO3": ("Some {A} are {B}.", "Some {C} are not {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "OI2": ("Some {B} are not {A}.", "Some {C} are {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "OI4": ("Some {B} are not {A}.", "Some {B} are {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "OO2": ("Some {B} are not {A}.", "Some {C} are not {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "AA1": ("All {A} are {B}.", "All {B} are {C}.", ["Some {C} are not {A}."]),
    "AO3": ("All {A} are {B}.", "Some {C} are not {B}.", ["All {A} are {C}." , "Some {A} are {C}.", "Some {C} are {A}."]),
    "IA1": ("Some {A} are {B}.", "All {B} are {C}.", ["All {A} are {C}." , "Some {C} are not {A}."]),
    "IA4": ("Some {B} are {A}.", "All {B} are {C}.", ["All {A} are {C}." , "Some {C} are not {A}."]),
    "OA4": ("Some {B} are not {A}.", "All {B} are {C}.", ["All {A} are {C}." , "Some {A} are {C}.", "Some {C} are {A}."]),
}

invalid_plausible_diff_chain_C = {
    "AO1": ("All {A} are {B}.", "Some {B} are not {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IO1": ("Some {A} are {B}.", "Some {B} are not {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IO2": ("Some {B} are {A}.", "Some {C} are not {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IO3": ("Some {A} are {B}.", "Some {C} are not {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IO4": ("Some {B} are {A}.", "Some {B} are not {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OE2": ("Some {B} are not {A}.", "No {C} are {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OE4": ("Some {B} are not {A}.", "No {B} are {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OO2": ("Some {B} are not {A}.", "Some {C} are not {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OO4": ("Some {B} are not {A}.", "Some {B} are not {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AO3": ("All {A} are {B}.", "Some {C} are not {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "IE1": ("Some {A} are {B}.", "No {B} are {C}.", ["No {A} are {C}.", "No {C} are {A}.","Some {C} are not {A}."]),
    "IE2": ("Some {B} are {A}.", "No {C} are {B}.", ["No {A} are {C}.", "No {C} are {A}.","Some {C} are not {A}."]),
    "IE3": ("Some {A} are {B}.", "No {C} are {B}.", ["No {A} are {C}.", "No {C} are {A}.","Some {C} are not {A}."]),
    "IE4": ("Some {B} are {A}.", "No {B} are {C}.", ["No {A} are {C}.", "No {C} are {A}.","Some {C} are not {A}."])
}

invalid_plausible_diff_chain_B = {
        "EE1": ("No {A} are {B}.", "No {B} are {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "EE2": ("No {B} are {A}.", "No {C} are {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "EE3": ("No {A} are {B}.", "No {C} are {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "EE4": ("No {B} are {A}.", "No {B} are {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "EO1": ("No {A} are {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "EO2": ("No {B} are {A}.", "Some {C} are not {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "EO3": ("No {A} are {B}.", "Some {C} are not {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "EO4": ("No {B} are {A}.", "Some {B} are not {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "OE1": ("Some {A} are not {B}.", "No {B} are {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "OE2": ("Some {B} are not {A}.", "No {C} are {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "OE3": ("Some {A} are not {B}.", "No {C} are {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "OE4": ("Some {B} are not {A}.", "No {B} are {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "OO1": ("Some {A} are not {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "OO2": ("Some {B} are not {A}.", "Some {C} are not {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "OO3": ("Some {A} are not {B}.", "Some {C} are not {B}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
        "OO4": ("Some {B} are not {A}.", "Some {B} are not {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."])
}

invalid_plausible_diff_chain_A = {
        "EO2": ("No {B} are {A}.", "Some {C} are not {B}.", ["Some {A} are not {C}.", "Some {C} are not {A}.", "No {A} are {C}.", "No {C} are {A}."]),
        "EO3": ("No {A} are {B}.", "Some {C} are not {B}.", ["Some {A} are not {C}.", "Some {C} are not {A}.", "No {A} are {C}.", "No {C} are {A}."]),
        "OA1": ("Some {A} are not {B}.", "All {B} are {C}.", ["Some {A} are not {C}.", "Some {C} are not {A}.", "No {A} are {C}.", "No {C} are {A}."]),
        "OI1": ("Some {A} are not {B}.", "Some {B} are {C}.", ["Some {A} are not {C}.", "Some {C} are not {A}.", "No {A} are {C}.", "No {C} are {A}."]),
        "OI2": ("Some {B} are not {A}.", "Some {C} are {B}.", ["Some {A} are not {C}.", "Some {C} are not {A}.", "No {A} are {C}.", "No {C} are {A}."]),
        "OI3": ("Some {A} are not {B}.", "Some {C} are {B}.", ["Some {A} are not {C}.", "Some {C} are not {A}.", "No {A} are {C}.", "No {C} are {A}."]),
        "OI4": ("Some {B} are not {A}.", "Some {B} are {C}.", ["Some {A} are not {C}.", "Some {C} are not {A}.", "No {A} are {C}.", "No {C} are {A}."]),
        "OO2": ("Some {B} are not {A}.", "Some {C} are not {B}.", ["Some {A} are not {C}.", "Some {C} are not {A}.", "No {A} are {C}.", "No {C} are {A}."]),
        "OO3": ("Some {A} are not {B}.", "Some {C} are not {B}.", ["Some {A} are not {C}.", "Some {C} are not {A}.", "No {A} are {C}.", "No {C} are {A}."]), 
        "OA4": ("Some {B} are not {A}.", "All {B} are {C}.", ["Some {A} are not {C}.", "No {A} are {C}.", "No {C} are {A}."]),
        "EA1": ("No {A} are {B}.", "All {B} are {C}.", ["Some {A} are not {C}.", "No {A} are {C}.", "No {C} are {A}."]),
        "EA4": ("No {B} are {A}.", "All {B} are {C}.", ["Some {A} are not {C}.", "No {A} are {C}.", "No {C} are {A}."]),
        "EI1": ("No {A} are {B}.", "Some {B} are {C}.", ["Some {A} are not {C}.", "No {A} are {C}.", "No {C} are {A}."]),
        "EI2": ("No {B} are {A}.", "Some {C} are {B}.", ["Some {A} are not {C}.", "No {A} are {C}.", "No {C} are {A}."]),
        "EI3": ("No {A} are {B}.", "Some {C} are {B}.", ["Some {A} are not {C}.", "No {A} are {C}.", "No {C} are {A}."]),
        "EI4": ("No {B} are {A}.", "Some {B} are {C}.", ["Some {A} are not {C}.", "No {A} are {C}.", "No {C} are {A}."])
}

invalid_plausible_diff_chain = {
        "EE1": ("No {A} are {B}.", "No {B} are {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "EE2": ("No {B} are {A}.", "No {C} are {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "EE3": ("No {A} are {B}.", "No {C} are {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "EE4": ("No {B} are {A}.", "No {B} are {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "EO1": ("No {A} are {B}.", "Some {B} are not {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "EO2": ("No {B} are {A}.", "Some {C} are not {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "EO3": ("No {A} are {B}.", "Some {C} are not {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "EO4": ("No {B} are {A}.", "Some {B} are not {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "OE1": ("Some {A} are not {B}.", "No {B} are {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "OE2": ("Some {B} are not {A}.", "No {C} are {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "OE3": ("Some {A} are not {B}.", "No {C} are {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "OE4": ("Some {B} are not {A}.", "No {B} are {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "OO1": ("Some {A} are not {B}.", "Some {B} are not {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "OO2": ("Some {B} are not {A}.", "Some {C} are not {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "OO3": ("Some {A} are not {B}.", "Some {C} are not {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
        "OO4": ("Some {B} are not {A}.", "Some {B} are not {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."])
}

valid_implausible_same_chain = {
    "AA2": ("All {B} are {A}.", "All {C} are {B}.",
            ["All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AA4": ("All {B} are {A}.", "All {B} are {C}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "AI2": ("All {B} are {A}.", "Some {C} are {B}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "AI4": ("All {B} are {A}.", "Some {B} are {C}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "AE1": ("All {A} are {B}.", "No {B} are {C}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AE2": ("All {B} are {A}.", "No {C} are {B}.",
            ["Some {A} are not {C}."]),
    "AE3": ("All {A} are {B}.", "No {C} are {B}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AE4": ("All {B} are {A}.", "No {B} are {C}.",
            ["Some {A} are not {C}."]),
    "AO4": ("All {B} are {A}.", "Some {B} are not {C}.",
            ["Some {A} are not {C}."]),
    "IE1": ("Some {A} are {B}.", "No {B} are {C}.",
            ["Some {A} are not {C}."]),
    "IE2": ("Some {B} are {A}.", "No {C} are {B}.",
            ["Some {A} are not {C}."]),
    "IE3": ("Some {A} are {B}.", "No {C} are {B}.",
            ["Some {A} are not {C}."]),
    "IE4": ("Some {B} are {A}.", "No {B} are {C}.",
            ["Some {A} are not {C}."]),
    "EA1": ("No {A} are {B}.", "All {B} are {C}.",
            ["Some {C} are not {A}."]),
    "EA2": ("No {B} are {A}.", "All {C} are {B}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EA3": ("No {A} are {B}.", "All {C} are {B}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EA4": ("No {B} are {A}.", "All {B} are {C}.",
            ["Some {C} are not {A}."]),
    "EI1": ("No {A} are {B}.", "Some {B} are {C}.",
            ["Some {C} are not {A}."]),
    "EI2": ("No {B} are {A}.", "Some {C} are {B}.",
            ["Some {C} are not {A}."]),
    "EI3": ("No {A} are {B}.", "Some {C} are {B}.",
            ["Some {C} are not {A}."]),
    "EI4": ("No {B} are {A}.", "Some {B} are {C}.",
            ["Some {C} are not {A}."]),
    "OA3": ("Some {A} are not {B}.", "All {C} are {B}.",
            ["Some {A} are not {C}."])
}

valid_implausible_diff_chain_A = {
    "AA1": ("All {A} are {B}.", "All {B} are {C}.",
                ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AA2": ("All {B} are {A}.", "All {C} are {B}.",
            ["All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AA4": ("All {B} are {A}.", "All {B} are {C}.",
            ["Some {A} are {C}, Some {C} are {A}."]),
    "AI2": ("All {B} are {A}.", "Some {C} are {B}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "AI4": ("All {B} are {A}.", "Some {B} are {C}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "AE1": ("All {A} are {B}.", "No {B} are {C}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AE2": ("All {B} are {A}.", "No {C} are {B}.",
            ["Some {A} are not {C}."]),
    "AE3": ("All {A} are {B}.", "No {C} are {B}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AE4": ("All {B} are {A}.", "No {B} are {C}.",
            ["Some {A} are not {C}."]),
    "AO3": ("All {A} are {B}.", "Some {C} are not {B}.",
            ["Some {C} are not {A}."]),
    "AO4": ("All {B} are {A}.", "Some {B} are not {C}.",
            ["Some {A} are not {C}."]),
    "IA1": ("Some {A} are {B}.", "All {B} are {C}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "IA4": ("Some {B} are {A}.", "All {B} are {C}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "IE1": ("Some {A} are {B}.", "No {B} are {C}.",
            ["Some {A} are not {C}."]),
    "IE2": ("Some {B} are {A}.", "No {C} are {B}.",
            ["Some {A} are not {C}."]),
    "IE3": ("Some {A} are {B}.", "No {C} are {B}.",
            ["Some {A} are not {C}."]),
    "IE4": ("Some {B} are {A}.", "No {B} are {C}.",
            ["Some {A} are not {C}."]),
    "EA2": ("No {B} are {A}.", "All {C} are {B}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EA3": ("No {A} are {B}.", "All {C} are {B}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OA3": ("Some {A} are not {B}.", "All {C} are {B}.",
            ["Some {A} are not {C}."]),
}

valid_implausible_diff_chain_B = {
    "AA1": ("All {A} are {B}.", "All {B} are {C}.",
                ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AA2": ("All {B} are {A}.", "All {C} are {B}.",
            ["All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AA4": ("All {B} are {A}.", "All {B} are {C}.",
            ["Some {A} are {C}, Some {C} are {A}."]),
    "AI2": ("All {B} are {A}.", "Some {C} are {B}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "AI4": ("All {B} are {A}.", "Some {B} are {C}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "AE1": ("All {A} are {B}.", "No {B} are {C}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AE2": ("All {B} are {A}.", "No {C} are {B}.",
            ["Some {A} are not {C}."]),
    "AE3": ("All {A} are {B}.", "No {C} are {B}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AE4": ("All {B} are {A}.", "No {B} are {C}.",
            ["Some {A} are not {C}."]),
    "AO3": ("All {A} are {B}.", "Some {C} are not {B}.",
            ["Some {C} are not {A}."]),
    "AO4": ("All {B} are {A}.", "Some {B} are not {C}.",
            ["Some {A} are not {C}."]),
    "IA1": ("Some {A} are {B}.", "All {B} are {C}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "IA4": ("Some {B} are {A}.", "All {B} are {C}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "IE1": ("Some {A} are {B}.", "No {B} are {C}.",
            ["Some {A} are not {C}."]),
    "IE2": ("Some {B} are {A}.", "No {C} are {B}.",
            ["Some {A} are not {C}."]),
    "IE3": ("Some {A} are {B}.", "No {C} are {B}.",
            ["Some {A} are not {C}."]),
    "IE4": ("Some {B} are {A}.", "No {B} are {C}.",
            ["Some {A} are not {C}."]),
    "EA1": ("No {A} are {B}.", "All {B} are {C}.",
            ["Some {C} are not {A}."]),
    "EA2": ("No {B} are {A}.", "All {C} are {B}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EA3": ("No {A} are {B}.", "All {C} are {B}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EA4": ("No {B} are {A}.", "All {B} are {C}.",
            ["Some {C} are not {A}."]),
    "EI1": ("No {A} are {B}.", "Some {B} are {C}.",
            ["Some {C} are not {A}."]),
    "EI2": ("No {B} are {A}.", "Some {C} are {B}.",
            ["Some {C} are not {A}."]),
    "EI3": ("No {A} are {B}.", "Some {C} are {B}.",
            ["Some {C} are not {A}."]),
    "EI4": ("No {B} are {A}.", "Some {B} are {C}.",
            ["Some {C} are not {A}."]),
    "OA3": ("Some {A} are not {B}.", "All {C} are {B}.",
            ["Some {A} are not {C}."]),
    "OA4": ("Some {B} are not {A}.", "All {B} are {C}.",
            ["Some {C} are not {A}."])
}

valid_implausible_diff_chain_C = {
    "AA1": ("All {A} are {B}.", "All {B} are {C}.",
                ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AA2": ("All {B} are {A}.", "All {C} are {B}.",
            ["All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AA4": ("All {B} are {A}.", "All {B} are {C}.",
            ["Some {A} are {C}, Some {C} are {A}."]),
    "AI2": ("All {B} are {A}.", "Some {C} are {B}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "AI4": ("All {B} are {A}.", "Some {B} are {C}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "IA1": ("Some {A} are {B}.", "All {B} are {C}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "IA4": ("Some {B} are {A}.", "All {B} are {C}.",
            ["Some {A} are {C}.", "Some {C} are {A}."]),
    "EA1": ("No {A} are {B}.", "All {B} are {C}.",
            ["Some {C} are not {A}."]),
    "EA2": ("No {B} are {A}.", "All {C} are {B}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EA3": ("No {A} are {B}.", "All {C} are {B}.",
            ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EA4": ("No {B} are {A}.", "All {B} are {C}.",
            ["Some {C} are not {A}."]),
    "EI1": ("No {A} are {B}.", "Some {B} are {C}.",
            ["Some {C} are not {A}."]),
    "EI2": ("No {B} are {A}.", "Some {C} are {B}.",
            ["Some {C} are not {A}."]),
    "EI3": ("No {A} are {B}.", "Some {C} are {B}.",
            ["Some {C} are not {A}."]),
    "EI4": ("No {B} are {A}.", "Some {B} are {C}.",
            ["Some {C} are not {A}."]),
    "OA3": ("Some {A} are not {B}.", "All {C} are {B}.",
            ["Some {A} are not {C}."]),
    "OA4": ("Some {B} are not {A}.", "All {B} are {C}.",
            ["Some {C} are not {A}."])
}

invalid_implausible_same_chain = {
    "AA3": ("All {A} are {B}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI1": ("All {A} are {B}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "AI3": ("All {A} are {B}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "AO1": ("All {A} are {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AO2": ("All {B} are {A}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "IA2": ("Some {B} are {A}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IA3": ("Some {A} are {B}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "II1": ("Some {A} are {B}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "II2": ("Some {B} are {A}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "II3": ("Some {A} are {B}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "II4": ("Some {B} are {A}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),

    "IO1": ("Some {A} are {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IO2": ("Some {B} are {A}.", "Some {C} are not {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "IO3": ("Some {A} are {B}.", "Some {C} are not {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "IO4": ("Some {B} are {A}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "EE1": ("No {A} are {B}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EE2": ("No {B} are {A}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EE3": ("No {A} are {B}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EE4": ("No {B} are {A}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "EO1": ("No {A} are {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EO2": ("No {B} are {A}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EO3": ("No {A} are {B}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EO4": ("No {B} are {A}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "OA1": ("Some {A} are not {B}.", "All {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OA2": ("Some {B} are not {A}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "OI1": ("Some {A} are not {B}.", "Some {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OI2": ("Some {B} are not {A}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "OI3": ("Some {A} are not {B}.", "Some {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OI4": ("Some {B} are not {A}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),

    "OE1": ("Some {A} are not {B}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OE2": ("Some {B} are not {A}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OE3": ("Some {A} are not {B}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OE4": ("Some {B} are not {A}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "OO1": ("Some {A} are not {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OO2": ("Some {B} are not {A}.", "Some {C} are not {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "OO3": ("Some {A} are not {B}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OO4": ("Some {B} are not {A}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AA2": ("All {B} are {A}.", "All {C} are {B}.",
            ["All {A} are {C}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AA4": ("All {B} are {A}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI2": ("All {B} are {A}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI4": ("All {B} are {A}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AE1": ("All {A} are {B}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AE2": ("All {B} are {A}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "AE3": ("All {A} are {B}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.","Some {A} are {C}.", "Some {C} are {A}."]),
    "AE4": ("All {B} are {A}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "AO4": ("All {B} are {A}.", "Some {B} are not {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IE1": ("Some {A} are {B}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IE2": ("Some {B} are {A}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IE3": ("Some {A} are {B}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IE4": ("Some {B} are {A}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "EA1": ("No {A} are {B}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EA2": ("No {B} are {A}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EA3": ("No {A} are {B}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.","Some {A} are {C}.", "Some {C} are {A}."]),
    "EA4": ("No {B} are {A}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EI1": ("No {A} are {B}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EI2": ("No {B} are {A}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EI3": ("No {A} are {B}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EI4": ("No {B} are {A}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "OA3": ("Some {A} are not {B}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."])
}

invalid_implausible_diff_chain_A = {
    "AA3": ("All {A} are {B}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI1": ("All {A} are {B}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "AI3": ("All {A} are {B}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "AO1": ("All {A} are {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AO2": ("All {B} are {A}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "IA2": ("Some {B} are {A}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IA3": ("Some {A} are {B}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "II1": ("Some {A} are {B}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "II2": ("Some {B} are {A}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "II3": ("Some {A} are {B}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "II4": ("Some {B} are {A}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),

    "IO1": ("Some {A} are {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IO2": ("Some {B} are {A}.", "Some {C} are not {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "IO3": ("Some {A} are {B}.", "Some {C} are not {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "IO4": ("Some {B} are {A}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "EE1": ("No {A} are {B}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EE2": ("No {B} are {A}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EE3": ("No {A} are {B}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EE4": ("No {B} are {A}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "EO1": ("No {A} are {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EO2": ("No {B} are {A}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EO3": ("No {A} are {B}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EO4": ("No {B} are {A}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "OA1": ("Some {A} are not {B}.", "All {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "OA2": ("Some {B} are not {A}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "OI1": ("Some {A} are not {B}.", "Some {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "OI2": ("Some {B} are not {A}.", "Some {C} are {B}.", ["All {C} are {A}."]),
    "OI3": ("Some {A} are not {B}.", "Some {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "OI4": ("Some {B} are not {A}.", "Some {B} are {C}.", ["All {C} are {A}."]),

    "OE1": ("Some {A} are not {B}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OE2": ("Some {B} are not {A}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OE3": ("Some {A} are not {B}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OE4": ("Some {B} are not {A}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "OO1": ("Some {A} are not {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OO2": ("Some {B} are not {A}.", "Some {C} are not {B}.", ["All {C} are {A}."]),
    "OO3": ("Some {A} are not {B}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "OO4": ("Some {B} are not {A}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AA1": ("All {A} are {B}.", "All {B} are {C}.",
                ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AA2": ("All {B} are {A}.", "All {C} are {B}.",
            ["All {A} are {C}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AA4": ("All {B} are {A}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI2": ("All {B} are {A}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI4": ("All {B} are {A}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AE1": ("All {A} are {B}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AE2": ("All {B} are {A}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "AE3": ("All {A} are {B}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.","Some {A} are {C}.", "Some {C} are {A}."]),
    "AE4": ("All {B} are {A}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "AO3": ("All {A} are {B}.", "Some {C} are not {B}.",
            ["Some {A} are not {C}.", "All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AO4": ("All {B} are {A}.", "Some {B} are not {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IA1": ("Some {A} are {B}.", "All {B} are {C}.",
            ["Some {A} are not {C}.", "Some {C} are not {A}.", "All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}."]),
    "IA4": ("Some {B} are {A}.", "All {B} are {C}.",
            ["Some {A} are not {C}.", "Some {C} are not {A}.", "All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}."]),
    "IE1": ("Some {A} are {B}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IE2": ("Some {B} are {A}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IE3": ("Some {A} are {B}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IE4": ("Some {B} are {A}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "EA1": ("No {A} are {B}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EA2": ("No {B} are {A}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EA3": ("No {A} are {B}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.","Some {A} are {C}.", "Some {C} are {A}."]),
    "EA4": ("No {B} are {A}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EI1": ("No {A} are {B}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EI2": ("No {B} are {A}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EI3": ("No {A} are {B}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EI4": ("No {B} are {A}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "OA3": ("Some {A} are not {B}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."])
}

invalid_implausible_diff_chain_B = {
    "AA3": ("All {A} are {B}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI1": ("All {A} are {B}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "AI3": ("All {A} are {B}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "AO1": ("All {A} are {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AO2": ("All {B} are {A}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "IA2": ("Some {B} are {A}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IA3": ("Some {A} are {B}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "II1": ("Some {A} are {B}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "II2": ("Some {B} are {A}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "II3": ("Some {A} are {B}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "II4": ("Some {B} are {A}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),

    "IO1": ("Some {A} are {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IO2": ("Some {B} are {A}.", "Some {C} are not {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "IO3": ("Some {A} are {B}.", "Some {C} are not {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "IO4": ("Some {B} are {A}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "EE1": ("No {A} are {B}.", "No {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "EE2": ("No {B} are {A}.", "No {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "EE3": ("No {A} are {B}.", "No {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "EE4": ("No {B} are {A}.", "No {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),

    "EO1": ("No {A} are {B}.", "Some {B} are not {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "EO2": ("No {B} are {A}.", "Some {C} are not {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "EO3": ("No {A} are {B}.", "Some {C} are not {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "EO4": ("No {B} are {A}.", "Some {B} are not {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),

    "OA1": ("Some {A} are not {B}.", "All {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OA2": ("Some {B} are not {A}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "OI1": ("Some {A} are not {B}.", "Some {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OI2": ("Some {B} are not {A}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "OI3": ("Some {A} are not {B}.", "Some {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OI4": ("Some {B} are not {A}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),

    "OE1": ("Some {A} are not {B}.", "No {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "OE2": ("Some {B} are not {A}.", "No {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "OE3": ("Some {A} are not {B}.", "No {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "OE4": ("Some {B} are not {A}.", "No {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),

    "OO1": ("Some {A} are not {B}.", "Some {B} are not {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "OO2": ("Some {B} are not {A}.", "Some {C} are not {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "OO3": ("Some {A} are not {B}.", "Some {C} are not {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "OO4": ("Some {B} are not {A}.", "Some {B} are not {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "AA1": ("All {A} are {B}.", "All {B} are {C}.",
                ["All {C} are {A}.","No {A} are {C}.","No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AA2": ("All {B} are {A}.", "All {C} are {B}.",
            ["All {A} are {C}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AA4": ("All {B} are {A}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI2": ("All {B} are {A}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI4": ("All {B} are {A}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AE1": ("All {A} are {B}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AE2": ("All {B} are {A}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "AE3": ("All {A} are {B}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.","Some {A} are {C}.", "Some {C} are {A}."]),
    "AE4": ("All {B} are {A}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "AO3": ("All {A} are {B}.", "Some {C} are not {B}.",
            ["All {A} are {C}.", "All {C} are {A}.","No {A} are {C}.", "No {C} are {A}.","Some {A} are {C}.", "Some {C} are {A}.","Some {A} are not {C}."]),
    "IA1": ("Some {A} are {B}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IA4": ("Some {B} are {A}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AO4": ("All {B} are {A}.", "Some {B} are not {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IE1": ("Some {A} are {B}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IE2": ("Some {B} are {A}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IE3": ("Some {A} are {B}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IE4": ("Some {B} are {A}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "EA1": ("No {A} are {B}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EA2": ("No {B} are {A}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EA3": ("No {A} are {B}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.","Some {A} are {C}.", "Some {C} are {A}."]),
    "EA4": ("No {B} are {A}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EI1": ("No {A} are {B}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EI2": ("No {B} are {A}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EI3": ("No {A} are {B}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EI4": ("No {B} are {A}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "OA3": ("Some {A} are not {B}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "OA4": ("Some {B} are not {A}.", "All {B} are {C}.",
            ["All {A} are {C}.","All {C} are {A}.","No {A} are {C}.","No {C} are {A}.","Some {A} are {C}.","Some {C} are {A}.","Some {A} are not {C}."])
}

invalid_implausible_diff_chain_C = {
    "AA3": ("All {A} are {B}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI1": ("All {A} are {B}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "AI3": ("All {A} are {B}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "AO1": ("All {A} are {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AO2": ("All {B} are {A}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "IA2": ("Some {B} are {A}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IA3": ("Some {A} are {B}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "II1": ("Some {A} are {B}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "II2": ("Some {B} are {A}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "II3": ("Some {A} are {B}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "II4": ("Some {B} are {A}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),

    "IO1": ("Some {A} are {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "IO2": ("Some {B} are {A}.", "Some {C} are not {B}.", ["All {C} are {A}."]),
    "IO3": ("Some {A} are {B}.", "Some {C} are not {B}.", ["All {C} are {A}."]),
    "IO4": ("Some {B} are {A}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),

    "EE1": ("No {A} are {B}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EE2": ("No {B} are {A}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EE3": ("No {A} are {B}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EE4": ("No {B} are {A}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "EO1": ("No {A} are {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EO2": ("No {B} are {A}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EO3": ("No {A} are {B}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "EO4": ("No {B} are {A}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "OA1": ("Some {A} are not {B}.", "All {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OA2": ("Some {B} are not {A}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),

    "OI1": ("Some {A} are not {B}.", "Some {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OI2": ("Some {B} are not {A}.", "Some {C} are {B}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "OI3": ("Some {A} are not {B}.", "Some {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OI4": ("Some {B} are not {A}.", "Some {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),

    "OE1": ("Some {A} are not {B}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OE2": ("Some {B} are not {A}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "OE3": ("Some {A} are not {B}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OE4": ("Some {B} are not {A}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),

    "OO1": ("Some {A} are not {B}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OO2": ("Some {B} are not {A}.", "Some {C} are not {B}.", ["All {C} are {A}."]),
    "OO3": ("Some {A} are not {B}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "OO4": ("Some {B} are not {A}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AA1": ("All {A} are {B}.", "All {B} are {C}.",
                ["All {C} are {A}.","No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AA2": ("All {B} are {A}.", "All {C} are {B}.",
            ["All {A} are {C}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AA4": ("All {B} are {A}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI2": ("All {B} are {A}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI4": ("All {B} are {A}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IA1": ("Some {A} are {B}.", "All {B} are {C}.",
            ["All {A} are {C}.","All {C} are {A}.", "No {A} are {C}.","No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IA4": ("Some {B} are {A}.", "All {B} are {C}.",
            ["All {A} are {C}.","All {C} are {A}.", "No {A} are {C}.","No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AE1": ("All {A} are {B}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AE2": ("All {B} are {A}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "AE3": ("All {A} are {B}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.","Some {A} are {C}.", "Some {C} are {A}."]),
    "AE4": ("All {B} are {A}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "AO4": ("All {B} are {A}.", "Some {B} are not {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "IE1": ("Some {A} are {B}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "IE2": ("Some {B} are {A}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "IE3": ("Some {A} are {B}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "IE4": ("Some {B} are {A}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EA1": ("No {A} are {B}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EA2": ("No {B} are {A}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EA3": ("No {A} are {B}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.","Some {A} are {C}.", "Some {C} are {A}."]),
    "EA4": ("No {B} are {A}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EI1": ("No {A} are {B}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EI2": ("No {B} are {A}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EI3": ("No {A} are {B}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "EI4": ("No {B} are {A}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}."]),
    "OA3": ("Some {A} are not {B}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "OA4": ("Some {B} are not {A}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.","No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.","Some {A} are not {C}."])
}

valid_implausible_diff_chain = {
   "AA1": ("All {A} are {B}.", "All {B} are {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}."]),
   "AA2": ("All {B} are {A}.", "All {C} are {B}.", ["All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
   "AA4": ("All {B} are {A}.", "All {B} are {C}.", ["Some {A} are {C}.", "Some {C} are {A}."]),
   "AI2": ("All {B} are {A}.", "Some {C} are {B}.",["Some {A} are {C}.", "Some {C} are {A}."]),
   "AI4": ("All {B} are {A}.", "Some {B} are {C}.", ["Some {A} are {C}.", "Some {C} are {A}."]),
   "AE1": ("All {A} are {B}.", "No {B} are {C}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
   "AE2": ("All {B} are {A}.", "No {C} are {B}.", ["Some {A} are not {C}."]),
   "AE3": ("All {A} are {B}.", "No {C} are {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
   "AE4": ("All {B} are {A}.", "No {B} are {C}.", ["Some {A} are not {C}."]),
   "AO3": ("All {A} are {B}.", "Some {C} are not {B}.", ["Some {C} are not {A}."]),
   "AO4": ("All {B} are {A}.", "Some {B} are not {C}.", ["Some {A} are not {C}."]),
   "IA1": ("Some {A} are {B}.", "All {B} are {C}.", ["Some {A} are {C}.", "Some {C} are {A}."]),
   "IA4": ("Some {B} are {A}.", "All {B} are {C}.", ["Some {A} are {C}.", "Some {C} are {A}."]),
   "IE1": ("Some {A} are {B}.", "No {B} are {C}.", ["Some {A} are not {C}."]),
   "IE2": ("Some {B} are {A}.", "No {C} are {B}.", ["Some {A} are not {C}."]),
   "IE3": ("Some {A} are {B}.", "No {C} are {B}.", ["Some {A} are not {C}."]),
   "IE4": ("Some {B} are {A}.", "No {B} are {C}.", ["Some {A} are not {C}."]),
   "EA1": ("No {A} are {B}.", "All {B} are {C}.", ["Some {C} are not {A}."]),
   "EA2": ("No {B} are {A}.", "All {C} are {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
   "EA3": ("No {A} are {B}.", "All {C} are {B}.", ["No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
   "EA4": ("No {B} are {A}.", "All {B} are {C}.", ["Some {C} are not {A}."]),
   "EI1": ("No {A} are {B}.", "Some {B} are {C}.", ["Some {C} are not {A}."]),
   "EI2": ("No {B} are {A}.", "Some {C} are {B}.", ["Some {C} are not {A}."]),
   "EI3": ("No {A} are {B}.", "Some {C} are {B}.", ["Some {C} are not {A}."]),
   "EI4": ("No {B} are {A}.", "Some {B} are {C}.", ["Some {C} are not {A}."]),
   "OA3": ("Some {A} are not {B}.", "All {C} are {B}.", ["Some {A} are not {C}."]),
   "OA4": ("Some {B} are not {A}.", "All {B} are {C}.", ["Some {C} are not {A}."])
}

invalid_implausible_diff_chain = {
    "AA1": ("All {A} are {B}.", "All {B} are {C}.", ["All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AA2": ("All {B} are {A}.", "All {C} are {B}.", ["All {A} are {C}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AA4": ("All {B} are {A}.", "All {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI2": ("All {B} are {A}.", "Some {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AI4": ("All {B} are {A}.", "Some {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "AE1": ("All {A} are {B}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AE2": ("All {B} are {A}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "AE3": ("All {A} are {B}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "AE4": ("All {B} are {A}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}."]),
    "AO3": ("All {A} are {B}.", "Some {C} are not {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "AO4": ("All {B} are {A}.", "Some {B} are not {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {C} are not {A}."]),
    "IA1": ("Some {A} are {B}.", "All {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IA4": ("Some {B} are {A}.", "All {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}."]),
    "IE1": ("Some {A} are {B}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {C} are not {A}."]),
    "IE2": ("Some {B} are {A}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {C} are not {A}."]),
    "IE3": ("Some {A} are {B}.", "No {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {C} are not {A}."]),
    "IE4": ("Some {B} are {A}.", "No {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {C} are not {A}."]),
    "EA1": ("No {A} are {B}.", "All {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "EA2": ("No {B} are {A}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EA3": ("No {A} are {B}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
    "EA4": ("No {B} are {A}.", "All {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "EI1": ("No {A} are {B}.", "Some {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "EI2": ("No {B} are {A}.", "Some {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "EI3": ("No {A} are {B}.", "Some {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "EI4": ("No {B} are {A}.", "Some {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "No {A} are {C}.", "No {C} are {A}.", "Some {A} are not {C}."]),
    "OA3": ("Some {A} are not {B}.", "All {C} are {B}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {C} are not {A}.", "No {A} are {C}.", "No {C} are {A}."]),
    "OA4": ("Some {B} are not {A}.", "All {B} are {C}.", ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "No {A} are {C}.", "No {C} are {A}."]),
    "AA3": ("All {A} are {B}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "AI1": ("All {A} are {B}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "AI3": ("All {A} are {B}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "AO1": ("All {A} are {B}.", "Some {B} are not {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "AO2": ("All {B} are {A}.", "Some {C} are not {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "IA2": ("Some {B} are {A}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "IA3": ("Some {A} are {B}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "II1": ("Some {A} are {B}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "II2": ("Some {B} are {A}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "II3": ("Some {A} are {B}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "II4": ("Some {B} are {A}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "IO1": ("Some {A} are {B}.", "Some {B} are not {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "IO2": ("Some {A} are {B}.", "Some {C} are not {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "IO3": ("Some {A} are {B}.", "Some {C} are not {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "IO4": ("Some {B} are {A}.", "Some {C} are not {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "OA1": ("Some {A} are not {B}.", "All {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "OA2": ("Some {B} are not {A}.", "All {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

     "OI1": ("Some {A} are not {B}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "OI2": ("Some {B} are not {A}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "OI3": ("Some {A} are not {B}.", "Some {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),

    "OI4": ("Some {B} are not {A}.", "Some {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}.",
             "Some {A} are not {C}.", "Some {C} are not {A}.",
             "No {A} are {C}.", "No {C} are {A}."]),
             #seperator
    "EE1": ("No {A} are {B}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "EE2": ("No {B} are {A}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "EE3": ("No {A} are {B}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "EE4": ("No {B} are {A}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "EO1": ("No {A} are {B}.", "Some {B} are not {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "EO2": ("No {B} are {A}.", "Some {C} are not {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "EO3": ("No {A} are {B}.", "Some {C} are not {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "EO4": ("No {B} are {A}.", "Some {B} are not {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "OE1": ("Some {A} are not {B}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "OE2": ("Some {B} are not {A}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "OE3": ("Some {A} are not {B}.", "No {C} are {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "OE4": ("Some {B} are not {A}.", "No {B} are {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "OO1": ("Some {A} are not {B}.", "Some {B} are not {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "OO2": ("Some {B} are not {A}.", "Some {C} are not {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "OO3": ("Some {A} are not {B}.", "Some {C} are not {B}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."]),

    "OO4": ("Some {B} are not {A}.", "Some {B} are not {C}.",
            ["All {A} are {C}.", "All {C} are {A}.",
             "Some {A} are {C}.", "Some {C} are {A}."])
}

languages = ["english"]

used_nouns = set()  # Track nouns used across syllogisms

for i in languages:

        all_data = []

        # --- Valid Plausible ---
        all_data += generate_syllogisms(valid_plausible_same_chain, "valid_p_same_chain", True, True, 584, used_nouns)
        all_data += generate_syllogisms(valid_plausible_diff_chain_C, "valid_p_diff_chain_C", True, True, 843, used_nouns)
        all_data += generate_syllogisms(valid_plausible_diff_chain_A, "valid_p_diff_chain_A", True, True, 454, used_nouns)

        # --- Invalid Plausible ---
        all_data += generate_syllogisms(invalid_plausible_same_chain, "invalid_p_same_chain", False, True, 356, used_nouns)
        all_data += generate_syllogisms(invalid_plausible_diff_chain_C, "invalid_p_diff_chain_C", False, True, 330, used_nouns)
        all_data += generate_syllogisms(invalid_plausible_diff_chain_B, "invalid_p_diff_chain_B", False, True, 414, used_nouns)
        all_data += generate_syllogisms(invalid_plausible_diff_chain_A, "invalid_p_diff_chain_A", False, True, 369, used_nouns)
        all_data += generate_syllogisms(invalid_plausible_diff_chain, "invalid_p_diff_chain", False, True, 414, used_nouns)

        # --- Valid Implausible ---
        all_data += generate_syllogisms(valid_implausible_same_chain, "valid_np_same_chain", True, False, 357, used_nouns)
        all_data += generate_syllogisms(valid_implausible_diff_chain_A, "valid_np_diff_chain_A", True, False, 367, used_nouns)
        all_data += generate_syllogisms(valid_implausible_diff_chain_B, "valid_np_diff_chain_B", True, False, 431, used_nouns)
        all_data += generate_syllogisms(valid_implausible_diff_chain_C, "valid_np_diff_chain_C", True, False, 285, used_nouns)
        all_data += generate_syllogisms(valid_implausible_diff_chain, "valid_np_diff_chain", True, False, 440, used_nouns)

        # # --- Invalid Implausible ---
        all_data += generate_syllogisms(invalid_implausible_same_chain, "invalid_np_same_chain", False, False, 389, used_nouns)
        all_data += generate_syllogisms(invalid_implausible_diff_chain_A, "invalid_np_diff_chain_A", False, False, 362, used_nouns)
        all_data += generate_syllogisms(invalid_implausible_diff_chain_B, "invalid_np_diff_chain_B", False, False, 360, used_nouns)
        all_data += generate_syllogisms(invalid_implausible_diff_chain_C, "invalid_np_diff_chain_C", False, False, 368, used_nouns)
        all_data += generate_syllogisms(invalid_implausible_diff_chain, "invalid_np_diff_chain", False, False, 400, used_nouns)

        # Save syllogisms to JSON
        with open(f"simple_4_var/english_4_var_simple.json", "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=4, ensure_ascii=False)

        # ---- Save shared unique nouns ----
        shared_nouns_path = "simple_4_var/shared_unique_nouns.json"

        existing_nouns = set()
        try:
                with open(shared_nouns_path, "r", encoding="utf-8") as f:
                        existing_nouns = set(json.load(f))
        except FileNotFoundError:
                pass

        all_unique_nouns = existing_nouns.union(used_nouns)

        with open(shared_nouns_path, "w", encoding="utf-8") as f:
                json.dump(sorted(list(all_unique_nouns)), f, indent=4, ensure_ascii=False)

        print(f"Added {len(used_nouns)} nouns")
        print(f"Total unique nouns in shared file: {len(all_unique_nouns)}")
        print(f"Generated {len(all_data)} syllogisms and saved")