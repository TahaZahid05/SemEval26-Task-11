import random
import time
import uuid
import json
from collections import defaultdict
import pandas as pd
from gibberish import Gibberish
import re

class SymbolicPool:
    def __init__(self):
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

    def get_four(self):
        """Pick a category and return 3 unique variables."""
        # Combining sub-lists for the selection
        latin = self.latin_lower + self.latin_upper
        greek = self.greek_lower + self.greek_upper
        
        # Select one of the three pools
        category = random.choice([latin, greek, self.pseudo])
        
        # Return 3 unique samples from the chosen category
        return random.sample(category, 4)

# --- Utility functions ---
def generate_syllogisms(schema, validity, plausibility, count, pool):
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
        A, B, C, D = pool.get_four()

        prem1, prem2, conclusions = schema[all_labels[curr_ind]]
        values_for_first = re.findall(r"\{(.*?)\}", prem1)
        values_for_second = re.findall(r"\{(.*?)\}", prem2)

        first_first, first_second = values_for_first
        second_first, second_second = values_for_second
        A_four, B_four_first = A, B
        B_four_second, C_four = B, C 

        if first_first == "A":
            A_four = f"{D} that are {A}"
        elif first_first == "B":
            B_four_first = f"{D} that are {B}"
        
        if second_first == "B":
            B_four_second = f"{D} that are {B}"
        elif second_first == "C":
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

    return results


# Naming convention
# x_y_z
# x = valid/invalid
# y = plausible/implausible
# z = different chain / different chain X / same chain

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

for i in valid_schemas:
     invalid_schemas[i] = (valid_schemas[i][0], valid_schemas[i][1], list(set(all_possible_conclusions) - set(valid_schemas[i][2])))

# Naming cocurr_indnvention
# x_y_z
# x = valid/invalid
# y = plausible/implausible
# z = different chain / different chain X / same chain




languages = ["english"]
symbolic_pool = SymbolicPool()

for i in languages:

        all_data = []

        all_data += generate_syllogisms(valid_schemas, True, "neutral", 6000, symbolic_pool)

        all_data += generate_syllogisms(invalid_schemas, False, "neutral", 6000, symbolic_pool)


        with open(f"simple_symbolic_4_var/english_4_var_simple_symbolic.json", "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=4, ensure_ascii=False)

        print(f"Generated {len(all_data)} syllogisms and saved")