import random
import uuid
import json
import random
import math
import re
from gibberish import Gibberish
from collections import defaultdict
import pandas as pd
from itertools import combinations

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
        
def shuffle_premises(all_premises, num_relevant = 2):
    
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

class SymbolicPool:
    def __init__(self):
        self.already_used = []
        # 1. Latin Pool
        self.latin_upper = [chr(i) for i in range(65, 91)] # A-Z
        self.latin_lower = [chr(i) for i in range(97, 123)] # a-z
        
        # 2. Greek Pool (Unicode)
        # Lowercase
        self.greek_lower = [
            "\u03B1", "\u03B2", "\u03B3", "\u03B4", "\u03B5", "\u03B6", "\u03B7", "\u03B8", 
            "\u03B9", "\u03BA", "\u03BB", "\u03BC", "\u03BD", "\u03BE", "\u03BF", "\u03C0", 
            "\u03C1", "\u03C3", "\u03C4", "\u03C5", "\u03C6", "\u03C7", "\u03C8", "\u03C9"
        ]

        # Uppercase
        self.greek_upper = [
            "\u0391", "\u0392", "\u0393", "\u0394", "\u0395", "\u0396", "\u0397", "\u0398", 
            "\u0399", "\u039A", "\u039B", "\u039C", "\u039D", "\u039E", "\u039F", "\u03A0", 
            "\u03A1", "\u03A3", "\u03A4", "\u03A5", "\u03A6", "\u03A7", "\u03A8", "\u03A9"
        ]
        
        # 3. Pseudo-words (As suggested by Bertolazzi et al., 2024)
        # In practice, you would load your 4k gibberish words here
        self.pseudo = Gibberish().generate_words(4000)


    def get_gibberish(self, num):
        """Pick a category and return num unique variables."""
        # Combining sub-lists for the selection
        latin = self.latin_lower + self.latin_upper
        greek = self.greek_lower + self.greek_upper
        
        # Select one of the three pools
        category = random.choice([latin, greek, self.pseudo])
        picked = random.sample(category, num)
        avilable = [allowed for allowed in picked if allowed not in self.already_used]
        while len(avilable) < num:
            picked = random.sample(category, num)
            avilable = [allowed for allowed in picked if allowed not in self.already_used]

        self.already_used.extend(picked)
        # Return 3 unique samples from the chosen category
        return picked
    
    def clear(self):
        self.already_used.clear()


def generate_syllogisms(schema, pool, validity, plausibility, count = 1):
    
    results = []

    curr_ind = 0
    inner_ind = 0

    all_labels = list(schema.keys())
    
    for _ in range(count):

        while True:
            
            # Capture the specific Syllogism Key (e.g., "AI1", "AE3")
            current_syllogism_key = all_labels[curr_ind]
            prem1, prem2, conclusions = schema[current_syllogism_key]
            conclusion = conclusions[inner_ind]
            type_concl = concl_type(conclusion)
            irrelevant_premises_list = irrelevant_premise_conclusion[type_concl]

            is_valid = validity

            A, B, C, D = pool.get_gibberish(4)
            
            relevant_premises = [prem1.format(A = A, B = B, C = C), prem2.format(A = A, B = B, C = C)]
            
            bool_a = True
            bool_b = True
            bool_c = True

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
                    sentence = template.format(A=random.choice([temp_A, A]), B=random.choice([temp_B, B]), C=random.choice([temp_C, C]))
                    if sentence not in relevant_premises:
                        type_0_pool.append(sentence)
            random.shuffle(type_0_pool)
            
            total_irr_premises = []

            N = pick_num_irrelevant()

            for _ in range(N):
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
                    unrelated_one = pool.get_gibberish(1)[0]
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
                    # Type 2: Both variables are unrelated
                    chosen_key = random.choice(list(all_premises.keys()))
                    picked_irr_premise = all_premises[chosen_key]
                    values = re.findall(r"\{(.*?)\}", picked_irr_premise)
                    if len(values) < 2: continue 
                    first, second = values
                    unrelated_one, unrelated_two, unrelated_chance = pool.get_gibberish(3)
                    unrelated_one_chance = f"{unrelated_chance} that are {unrelated_one}"
                    temp_dict = {first: random.choice([unrelated_one,unrelated_one_chance]), second: unrelated_two}
                    candidate = f"{picked_irr_premise.format(**temp_dict)}"
                    total_irr_premises.append(candidate)
                        

            final_all_prems = relevant_premises + total_irr_premises
            shuffled_premises, relevant_indices = shuffle_premises(final_all_prems, num_relevant = 2)
            total_syllogism = " ".join(shuffled_premises + [conclusion.format(A = A, B = B, C = C)])

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

            pool.clear()

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



valid_schemas = {
     "AA1": ("All {A} are {B}.", "All {B} are {C}.", ["All {A} are {C}.", "Some {A} are {C}.", "Some {C} are {A}."]),
     "AA2": ("All {B} are {A}.", "All {C} are {B}.", ["All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}."]),
     "AA4": ("All {B} are {A}.", "All {B} are {C}.", ["Some {A} are {C}.", "Some {C} are {A}."]),
     "AI2": ("All {B} are {A}.", "Some {C} are {B}.", ["Some {A} are {C}.", "Some {C} are {A}."]),
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
     "IE2": ("Some {B} are {A}.", "No {B} are {C}.", ["Some {A} are not {C}."]),
     "IE3": ("Some {A} are {B}.", "No {C} are {B}.", ["Some {A} are not {C}."]),
     "IE4": ("Some {B} are {A}.", "No {C} are {B}.", ["Some {A} are not {C}."]),
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



all_possible_conclusions = ["All {A} are {C}.", "All {C} are {A}.", "Some {A} are {C}.", "Some {C} are {A}.", "Some {A} are not {C}.", "Some {C} are not {A}.", "No {A} are {C}.", "No {C} are {A}."]


invalid_schemas = {
     "AA3": ("All {A} are {B}.", "All {C} are {B}.", all_possible_conclusions),
     "AI1": ("All {A} are {B}.", "Some {B} are {C}.", all_possible_conclusions),
     "AI3": ("All {A} are {B}.", "Some {C} are {B}.", all_possible_conclusions),
     "AO1": ("All {A} are {B}.", "Some {B} are not {C}.", all_possible_conclusions),
     "AO2": ("All {B} are {A}.", "Some {C} are not {B}.", all_possible_conclusions),
     "IA2": ("Some {B} are {A}.", "All {C} are {B}.", all_possible_conclusions),
     "IA3": ("Some {A} are {B}.", "All {C} are {B}.", all_possible_conclusions),
     "II1": ("Some {A} are {B}.", "Some {B} are {C}.", all_possible_conclusions),
     "II2": ("Some {B} are {A}.", "Some {C} are {B}.", all_possible_conclusions),
     "II3": ("Some {A} are {B}.", "Some {C} are {B}.", all_possible_conclusions),
     "II4": ("Some {B} are {A}.", "Some {B} are {C}.", all_possible_conclusions),
     "IO1": ("Some {A} are {B}.", "Some {B} are not {C}.", all_possible_conclusions),
     "IO2": ("Some {B} are {A}.", "Some {C} are not {B}.", all_possible_conclusions),
     "IO3": ("Some {A} are {B}.", "Some {C} are not {B}.", all_possible_conclusions),
     "IO4": ("Some {B} are {A}.", "Some {B} are not {C}.", all_possible_conclusions),
     "EE1": ("No {A} are {B}.", "No {B} are {C}.", all_possible_conclusions),
     "EE2": ("No {B} are {A}.", "No {C} are {B}.", all_possible_conclusions),
     "EE3": ("No {A} are {B}.", "No {C} are {B}.", all_possible_conclusions),
     "EE4": ("No {B} are {A}.", "No {B} are {C}.", all_possible_conclusions),
     "EO1": ("No {A} are {B}.", "Some {B} are not {C}.", all_possible_conclusions),
     "EO2": ("No {B} are {A}.", "Some {C} are not {B}.", all_possible_conclusions),
     "EO3": ("No {A} are {B}.", "Some {C} are not {B}.", all_possible_conclusions),
     "EO4": ("No {B} are {A}.", "Some {B} are not {C}.", all_possible_conclusions),
     "OA1": ("Some {A} are not {B}.", "All {B} are {C}.", all_possible_conclusions),
     "OA2": ("Some {B} are not {A}.", "All {C} are {B}.", all_possible_conclusions),
     "OI1": ("Some {A} are not {B}.", "Some {B} are {C}.", all_possible_conclusions),
     "OI2": ("Some {B} are not {A}.", "Some {C} are {B}.", all_possible_conclusions),
     "OI3": ("Some {A} are not {B}.", "Some {C} are {B}.", all_possible_conclusions),
     "OI4": ("Some {B} are not {A}.", "Some {B} are {C}.", all_possible_conclusions),
     "OE1": ("Some {A} are not {B}.", "No {B} are {C}.", all_possible_conclusions),
     "OE2": ("Some {B} are not {A}.", "No {C} are {B}.", all_possible_conclusions),
     "OE3": ("Some {A} are not {B}.", "No {C} are {B}.", all_possible_conclusions),
     "OE4": ("Some {B} are not {A}.", "No {B} are {C}.", all_possible_conclusions),
     "OO1": ("Some {A} are not {B}.", "Some {B} are not {C}.", all_possible_conclusions),
     "OO2": ("Some {B} are not {A}.", "Some {C} are not {B}.", all_possible_conclusions),
     "OO3": ("Some {A} are not {B}.", "Some {C} are not {B}.", all_possible_conclusions),
     "OO4": ("Some {B} are not {A}.", "Some {B} are not {C}.", all_possible_conclusions),
}

# Add valid schema keys to invalid_schemas with complementary conclusions
for key, (prem1, prem2, valid_conclusions) in valid_schemas.items():
    if key not in invalid_schemas:
        invalid_conclusion_list = [c for c in all_possible_conclusions if c not in valid_conclusions]
        invalid_schemas[key] = (prem1, prem2, invalid_conclusion_list)


symbolic_pool = SymbolicPool()

languages = ["english", "german", "spanish", "french", "italian", "dutch", "portuguese", "russian", "chinese", "swahili", "bengali", "telugu"]

for i in languages:

        all_data = []

        # all_data += generate_syllogisms(valid_schemas, symbolic_pool, True, "neutral", 375)

        all_data += generate_syllogisms(valid_schemas, symbolic_pool, True, "neutral", 2254)

        all_data += generate_syllogisms(invalid_schemas, symbolic_pool, False, "neutral", 1879)


        with open(f"st4_{i}_simple_symbolic.json", "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=4, ensure_ascii=False)

        print(f"Generated {len(all_data)} syllogisms and saved")