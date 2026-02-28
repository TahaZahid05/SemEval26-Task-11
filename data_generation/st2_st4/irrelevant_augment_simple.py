import random
import time
import uuid
import json
import nltk
import random
import math
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
from itertools import combinations
p = inflect.engine()

FREQUENCY_THRESHOLD = 5

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


def concl_type(concl):
    x = concl.split(" ")

    if (x[0] == "All"):
        if (x[1] == "{A}"):
            return "Aac"
        else:
            return "Aca"
    elif (x[0] == "Some" and x[3] != "not"):
        if (x[1] == "{A}"):
            return "Iac"
        else:
            return "Ica"
    elif (x[0] == "No"):
        if (x[1] == "{A}"):
            return "Eac"
        else:
            return "Eca"
    else:
        if (x[1] == "{A}"):
            return "Oac"
        else:
            return "Oca"  
        
def shuffle_premises(all_premises, num_relevant=2):
    """
    Shuffle all premises while keeping track of which ones are relevant.
    Returns: (shuffled_premises, indices_of_relevant_premises)
    """
    # Create list of (premise, is_relevant) tuples
    premises_with_info = []
    for i, premise in enumerate(all_premises):
        is_relevant = (i < num_relevant)  # First num_relevant premises are the relevant ones
        premises_with_info.append((premise, is_relevant, i))
    
    # Shuffle the list
    random.shuffle(premises_with_info)
    
    # Extract shuffled premises and find new indices of relevant ones
    shuffled_premises = []
    relevant_indices = []
    
    for new_index, (premise, is_relevant, original_index) in enumerate(premises_with_info):
        shuffled_premises.append(premise)
        if is_relevant:
            relevant_indices.append(new_index)
    
    # Sort the relevant indices for consistency
    relevant_indices.sort()
    
    return shuffled_premises, relevant_indices


# def pick_num_irrelevant():
#     return random.choice([6, 7, 8])

def pick_num_irrelevant(max_irrelevant=6, lam=0.3):
    values = list(range(0, max_irrelevant + 1))
    
    # exponential decay weights: highest for 1, lowest for max_irrelevant
    weights = [math.exp(-lam * i) for i in values]
    
    # pick one value
    return random.choices(values, weights=weights, k=1)[0]

def generate_syllogisms(schema, chain_type, validity, plausibility, count=1, used_nouns_tracker=None):
    results = []
    curr_ind = 0
    inner_ind = 0
    all_labels = list(schema.keys())
    
    # Initialize tracker if not provided
    if used_nouns_tracker is None:
        used_nouns_tracker = set()
    
    for _ in range(count):
        while True:
            base_synset = random.choice(ALL_NOUN_SYNSETS)
            related_chain_clean = []
            curr_used = []

            # Capture the specific Syllogism Key (e.g., "AI1", "AE3")
            current_syllogism_key = all_labels[curr_ind]
            
            prem1, prem2, conclusions = schema[current_syllogism_key]
            conclusion = conclusions[inner_ind]
            type_concl = concl_type(conclusion)
            irrelevant_premises_list = irrelevant_premise_conclusion[type_concl]

            is_valid = True if chain_type.startswith("valid") else False

            # --- CHAIN GENERATION ---
            if chain_type in ["valid_p_same_chain", "invalid_p_same_chain", "valid_np_same_chain", "invalid_np_same_chain"]:
                    chain = get_hypernym_chain(base_synset, depth=2)
                    if len(chain) < 3: continue
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
            elif chain_type in ["valid_p_diff_chain_C", "invalid_p_diff_chain_C", "valid_np_diff_chain_C", "invalid_np_diff_chain_C"]:
                    chain = get_hypernym_chain(base_synset, depth=1)
                    if len(chain) < 2: continue
                    unrelated_syn = get_unrelated_noun(chain)
                    A, B, C = map(wn_name, chain + [unrelated_syn])
                    curr_used = chain + [unrelated_syn]
                    related_chain_clean.append(A)
                    related_chain_clean.append(B)
                    # D is the superset (hypernym) of all words in the chain
                    if chain[-1].hypernyms():
                        D = wn_name(chain[-1].hypernyms()[0])
                    else:
                        continue
            elif chain_type in ["invalid_p_diff_chain_B", "valid_np_diff_chain_B", "invalid_np_diff_chain_B"]:
                    chain = get_hypernym_chain(base_synset, depth=1)
                    if len(chain) < 2: continue
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
            elif chain_type in ["invalid_p_diff_chain_A", "valid_np_diff_chain_A", "invalid_np_diff_chain_A", "valid_p_diff_chain_A"]:
                    chain = get_hypernym_chain(base_synset, depth=1)
                    if len(chain) < 2: continue
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
            elif chain_type in ["valid_np_diff_chain", "invalid_p_diff_chain", "invalid_np_diff_chain"]:
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
            
            A, B, C, D = p.plural(A), p.plural(B), p.plural(C), p.plural(D)

            # Track nouns used in this syllogism
            used_nouns_tracker.add(A)
            used_nouns_tracker.add(B)
            used_nouns_tracker.add(C)
            used_nouns_tracker.add(D)

            relevant_premises = [prem1.format(A=A, B=B, C=C), prem2.format(A=A, B=B, C=C)]
            
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


            # --- (Keep your optimized Type 0/1/2 logic here from previous step) ---
            type_0_pool = []
            for key in irrelevant_premises_list:
                # Get the template and extract variables
                template = all_premises[key]
                template_values = re.findall(r"\{(.*?)\}", template)
                if len(template_values) >= 2:
                    first_var = template_values[0]
                    # Apply 4-variable conversion to first variable if applicable
                    temp_A, temp_B, temp_C = A, B, C
                    if first_var == "A" and bool_a:
                        temp_A = f"{D} that are {A}"
                    elif first_var == "B" and bool_b:
                        temp_B = f"{D} that are {B}"
                    elif first_var == "C" and bool_c:
                        temp_C = f"{D} that are {C}"
                    sentence = template.format(A=random.choice([temp_A,A]), B=random.choice([temp_B,B]), C=random.choice([temp_C,C]))
                    if sentence not in relevant_premises:
                        type_0_pool.append(sentence)
            random.shuffle(type_0_pool)
            
            total_irr_premises = []
            N = pick_num_irrelevant()

            attempts = 0
            MAX_ATTEMPTS = 50 * N

            while len(total_irr_premises) < N and attempts < MAX_ATTEMPTS:
                attempts += 1

                available_types = [1, 2]
                if len(type_0_pool) > 0:
                        available_types.append(0)

                types = random.choice(available_types)
                candidate = ""

                
                if types == 0:
                    candidate = type_0_pool.pop()
                    total_irr_premises.append(candidate)
                elif types == 1:
                    # Type 1: One variable from A/B/C, one unrelated
                    chosen_key = random.choice(list(all_premises.keys()))
                    picked_irr_premise = all_premises[chosen_key]
                    values = re.findall(r"\{(.*?)\}", picked_irr_premise)
                    if len(values) < 2: continue 
                    first, second = values
                    unrelated_one = p.plural(wn_name(get_unrelated_noun(curr_used)))
                    used_nouns_tracker.add(unrelated_one)
                    sub_random = random.randint(0, 1)
                    temp_dict = {}
                    if sub_random == 0:
                        val = C
                        val_second = C
                        if first == "A": 
                            val = A
                            val_second = A
                        elif first == "B": 
                            val = B
                            val_second = B
                        # Apply 4-variable conversion to first variable if it's from A/B/C
                        if first == "A" and bool_a:
                            val_second = f"{D} that are {A}"
                        elif first == "B" and bool_b:
                            val_second = f"{D} that are {B}"
                        elif first == "C" and bool_c:
                            val_second = f"{D} that are {C}"
                        temp_dict = {first: random.choice([val,val_second]), second: unrelated_one}
                    else:
                        val = C
                        if second == "A": val = A
                        elif second == "B": val = B
                        temp_dict = {first: unrelated_one, second: val}
                    candidate = f"{picked_irr_premise.format(**temp_dict)}"
                    total_irr_premises.append(candidate)
                else: 
                    # Type 2: Both variables are unrelated, introduce 4th variable for both
                    chosen_key = random.choice(list(all_premises.keys()))
                    picked_irr_premise = all_premises[chosen_key]
                    values = re.findall(r"\{(.*?)\}", picked_irr_premise)
                    if len(values) < 2: continue 
                    first, second = values
                    related_binary = random.randint(0, 1)
                    unrelated_one, unrelated_two = None, None
                    unrelated_one_chance = None
                    D_unrel_1, D_unrel_2 = None, None
                    
                    if related_binary:
                        # Both nouns are related to each other, introduce a 4th variable for them
                        base_unrel = get_unrelated_noun(curr_used)
                        unrel_chain = get_hypernym_chain(base_unrel, 1)
                        unrelated_one = p.plural(wn_name(unrel_chain[0]))
                        unrelated_two = p.plural(wn_name(unrel_chain[1]))
                        used_nouns_tracker.add(unrelated_one)
                        used_nouns_tracker.add(unrelated_two)
                        # Get a hypernym for the unrelated chains
                        if unrel_chain[-1].hypernyms():
                            D_unrel_1 = p.plural(wn_name(unrel_chain[-1].hypernyms()[0]))
                            used_nouns_tracker.add(D_unrel_1)
                        
                        D_unrel_2 = p.plural(wn_name(get_unrelated_noun(unrel_chain)))
                        used_nouns_tracker.add(D_unrel_2)

                        if D_unrel_1 is None:
                            D_unrel_1 = D_unrel_2
                        # Apply 4-variable only to first variable
                        unrelated_one_chance = f"{random.choice([D_unrel_1, D_unrel_2])} that are {unrelated_one}"
                    else:
                        # Nouns are completely unrelated, introduce separate 4th variables
                        unrelated_one_synset = get_unrelated_noun(curr_used)
                        unrel_chain = curr_used + [unrelated_one_synset]
                        unrelated_one = p.plural(wn_name(unrelated_one_synset))
                        unrelated_two = p.plural(wn_name(get_unrelated_noun(unrel_chain)))
                        used_nouns_tracker.add(unrelated_one)
                        used_nouns_tracker.add(unrelated_two)
                        # Get hypernyms for both unrelated nouns
                        if unrelated_one_synset.hypernyms():
                            D_unrel_1 = p.plural(wn_name(unrelated_one_synset.hypernyms()[0]))
                            used_nouns_tracker.add(D_unrel_1)

                        D_unrel_2 = p.plural(wn_name(get_unrelated_noun(unrel_chain)))
                        used_nouns_tracker.add(D_unrel_2)

                        if D_unrel_1 is None:
                            D_unrel_1 = D_unrel_2

                        unrelated_one_chance = f"{random.choice([D_unrel_1, D_unrel_2])} that are {unrelated_one}"
                    
                    temp_dict = {first: random.choice([unrelated_one,unrelated_one_chance]), second: unrelated_two}
                    candidate = f"{picked_irr_premise.format(**temp_dict)}"
                    total_irr_premises.append(candidate)

            final_all_prems = relevant_premises + total_irr_premises
            shuffled_premises, relevant_indices = shuffle_premises(final_all_prems, num_relevant=2)
            total_syllogism = " ".join(shuffled_premises + [conclusion.format(A=A, B=B, C=C)])

            # Determine which premises to include in the result
            if is_valid:
                premises_to_include = relevant_indices
            else:
                # For invalid syllogisms, check if premises are actually relevant to conclusion
                premises_to_include = []
                # Map premise templates back to keys and check relevance
                for i, premise_template in enumerate([prem1, prem2]):
                    for key, template in all_premises.items():
                        if template == premise_template:
                            # Check if this premise key is relevant for this conclusion type
                            if key in relevant_premises_conclusion[type_concl]:
                                # Include the index of this premise in shuffled_premises
                                premises_to_include.append(relevant_indices[i])
                            break

            results.append({
                "id": str(uuid.uuid4()),
                "syllogism": total_syllogism,
                "validity": validity,
                "plausibility": plausibility,
                "premises": premises_to_include
            })

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

all_premises = {
    "Aab": "All {A} are {B}.",
    "Aba": "All {B} are {A}.",
    "Iab": "Some {A} are {B}.",
    "Iba": "Some {B} are {A}.",
    "Oab": "Some {A} are not {B}.",
    "Oba": "Some {B} are not {A}.",
    "Eab": "No {A} are {B}.",
    "Eba": "No {B} are {A}.",
    "Abc": "All {B} are {C}.",
    "Acb": "All {C} are {B}.",
    "Ibc": "Some {B} are {C}.",
    "Icb": "Some {C} are {B}.",
    "Obc": "Some {B} are not {C}.",
    "Ocb": "Some {C} are not {B}.",
    "Ebc": "No {B} are {C}.",
    "Ecb": "No {C} are {B}.",
    "Aac": "All {A} are {C}.",
    "Aca": "All {C} are {A}.",
    "Iac": "Some {A} are {C}.",
    "Ica": "Some {C} are {A}.",
    "Eac": "No {A} are {C}.",
    "Eca": "No {C} are {A}.",
    "Oac": "Some {A} are not {C}.",
    "Oca": "Some {C} are not {A}."
}

relevant_premises_conclusion = {
    "Aac": ["Aab", "Abc", "Aac"],
    "Aca": ["Aba", "Acb", "Aca"],
    "Iac": ["Aab", "Abc", "Aba", "Acb", "Icb", "Ibc", "Iab", "Iba", "Iac"],
    "Ica": ["Aab", "Abc", "Aba", "Acb", "Icb", "Ibc", "Iab", "Iba", "Ica"],
    "Oac": ["Aab", "Ebc", "Ecb", "Aba", "Obc", "Iab", "Iba", "Eba", "Acb", "Eab", "Oab", "Oac"],
    "Oca": ["Aab", "Ebc", "Ecb", "Ocb", "Eab", "Abc", "Eba", "Acb", "Ibc", "Icb", "Oba", "Oca"],
    "Eac": ["Aab", "Ebc", "Ecb", "Eba", "Acb", "Eab", "Eac"],
    "Eca": ["Aab", "Ebc", "Ecb", "Eba", "Acb", "Eab", "Eca"]
}

irrelevant_premise_conclusion = {}

for i in relevant_premises_conclusion:
    irrelevant_premise_conclusion[i] = []
    for k in all_premises:
        if k not in relevant_premises_conclusion[i]:
            irrelevant_premise_conclusion[i].append(k)

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

# English (en), German (de), Spanish (es), French (fr), Italian (it), Dutch (nl), Portuguese (pt), Russian (ru), Chinese (zh), Swahili (sw), Bengali (bn), Telugu (te)

languages = ["english", "german", "spanish", "french", "italian", "dutch", "portuguese", "russian", "chinese", "swahili", "bengali", "telugu"]

for i in languages:

        all_data = []
        used_nouns = set()  # Track all unique nouns used

        # Remove One Premise:

        # # --- Valid Plausible ---
        # all_data += generate_syllogisms(valid_plausible_same_chain, "valid_p_same_chain", True, True, 116, used_nouns)
        # all_data += generate_syllogisms(valid_plausible_diff_chain_C, "valid_p_diff_chain_C", True, True, 168, used_nouns)
        # all_data += generate_syllogisms(valid_plausible_diff_chain_A, "valid_p_diff_chain_A", True, True, 90, used_nouns)

        # # --- Valid Implausible ---
        # all_data += generate_syllogisms(valid_implausible_same_chain, "valid_np_same_chain", True, False, 71, used_nouns)
        # all_data += generate_syllogisms(valid_implausible_diff_chain_A, "valid_np_diff_chain_A", True, False, 73, used_nouns)
        # all_data += generate_syllogisms(valid_implausible_diff_chain_B, "valid_np_diff_chain_B", True, False, 85, used_nouns)
        # all_data += generate_syllogisms(valid_implausible_diff_chain_C, "valid_np_diff_chain_C", True, False, 56, used_nouns)
        # all_data += generate_syllogisms(valid_implausible_diff_chain, "valid_np_diff_chain", True, False, 87, used_nouns)

        # Normal:

        # # --- Valid Plausible ---
        all_data += generate_syllogisms(valid_plausible_same_chain, "valid_p_same_chain", True, True, 699)
        all_data += generate_syllogisms(valid_plausible_diff_chain_C, "valid_p_diff_chain_C", True, True, 1010)
        all_data += generate_syllogisms(valid_plausible_diff_chain_A, "valid_p_diff_chain_A", True, True, 544, used_nouns)

        # # --- Invalid Plausible ---
        all_data += generate_syllogisms(invalid_plausible_same_chain, "invalid_p_same_chain", False, True, 355, used_nouns)
        all_data += generate_syllogisms(invalid_plausible_diff_chain_C, "invalid_p_diff_chain_C", False, True, 329, used_nouns)
        all_data += generate_syllogisms(invalid_plausible_diff_chain_B, "invalid_p_diff_chain_B", False, True, 413, used_nouns)
        all_data += generate_syllogisms(invalid_plausible_diff_chain_A, "invalid_p_diff_chain_A", False, True, 368, used_nouns)
        all_data += generate_syllogisms(invalid_plausible_diff_chain, "invalid_p_diff_chain", False, True, 413, used_nouns)

        # # --- Valid Implausible ---
        all_data += generate_syllogisms(valid_implausible_same_chain, "valid_np_same_chain", True, False, 428, used_nouns)
        all_data += generate_syllogisms(valid_implausible_diff_chain_A, "valid_np_diff_chain_A", True, False, 439, used_nouns)
        all_data += generate_syllogisms(valid_implausible_diff_chain_B, "valid_np_diff_chain_B", True, False, 516, used_nouns)
        all_data += generate_syllogisms(valid_implausible_diff_chain_C, "valid_np_diff_chain_C", True, False, 340, used_nouns)
        all_data += generate_syllogisms(valid_implausible_diff_chain, "valid_np_diff_chain", True, False, 527, used_nouns)

        # # --- Invalid Implausible ---
        all_data += generate_syllogisms(invalid_implausible_same_chain, "invalid_np_same_chain", False, False, 389, used_nouns)
        all_data += generate_syllogisms(invalid_implausible_diff_chain_A, "invalid_np_diff_chain_A", False, False, 362, used_nouns)
        all_data += generate_syllogisms(invalid_implausible_diff_chain_B, "invalid_np_diff_chain_B", False, False, 360, used_nouns)
        all_data += generate_syllogisms(invalid_implausible_diff_chain_C, "invalid_np_diff_chain_C", False, False, 368, used_nouns)
        all_data += generate_syllogisms(invalid_implausible_diff_chain, "invalid_np_diff_chain", False, False, 400, used_nouns)

        with open(f"st4_{i}_simple_one.json", "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=4, ensure_ascii=False)

        # Save unique nouns to a shared JSON file
        shared_nouns_path = f"shared_unique_nouns.json"
        
        # Load existing nouns if file exists
        existing_nouns = set()
        try:
            with open(shared_nouns_path, "r", encoding="utf-8") as f:
                existing_nouns = set(json.load(f))
        except FileNotFoundError:
            pass
        
        # Combine with new nouns
        all_unique_nouns = existing_nouns.union(used_nouns)
        
        # Save back to shared file
        with open(shared_nouns_path, "w", encoding="utf-8") as f:
                json.dump(sorted(list(all_unique_nouns)), f, indent=4, ensure_ascii=False)

        print(f"Generated {len(all_data)} syllogisms and saved")
        print(f"Added {len(used_nouns)} nouns, total unique nouns in shared file: {len(all_unique_nouns)}")