import json
from pathlib import Path

def merge_to_complete_data():
    base_dir = Path(__file__).parent
    
    # Source folders
    folders = ["SimpleData", "ComplexData"]
    output_dir = base_dir / "St4CompleteData"
    output_file = output_dir / "complete_dataset.json"
    
    # Create the output directory
    output_dir.mkdir(exist_ok=True)

    all_data = []

    for folder_name in folders:
        target_dir = base_dir / folder_name
        
        if not target_dir.exists():
            print(f"Folder {folder_name} not found. Skipping.")
            continue

        # Get all JSON files in the current folder
        json_files = list(target_dir.glob("*.json"))
        print(f"Reading {len(json_files)} files from {folder_name}...")

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    # If the file is a single object instead of a list
                    all_data.append(data)
                    
            except Exception as e:
                print(f"  Error reading {json_file.name}: {e}")

    # Save the master list to the new directory
    print(f"\nSaving {len(all_data)} total data points to {output_file.name}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

    print(f"Success! Complete dataset saved in {output_dir}")

if __name__ == "__main__":
    merge_to_complete_data()