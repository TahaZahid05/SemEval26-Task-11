# SemEval-2026 Task 11 - HABIB_TAZ

This repository contains the training scripts, data generation pipelines, and relevant documentation for our team's participation in **SemEval-2026 Task 11**.

## Repository Structure

The repository is organized into the following main directories:

- **`data_generation/`**: Contains scripts, notebooks, and pipelines to generate both simple and complex synthetic syllogisms across multiple languages.
  - **`st1_st3/`**: Data generation scripts targeting Subtask 1 and Subtask 3.
  - **`st2_st4/`**: Data generation scripts and Jupyter Notebooks targeting Subtask 2 and Subtask 4, including generation in multiple languages (e.g., French, German, Spanish, Bengali, Swahili, etc.).
- **`training_scripts/`**: Contains the model training environments for the different subtasks. 
  - `st1.ipynb`
  - `st2.ipynb`
  - `st3.ipynb`
  - `st4.py`
- **`paper/`**: Contains related documentation or paper materials for the submission.

## Data Generation Guidelines

Inside the `data_generation/st1_st3` and `data_generation/st2_st4` directories, you will find PDF guideline documents (`ST3_Data_Augmentation_Pipeline.pdf` and `ST4_Data_Augmentation_Pipeline.pdf`). 

**Important Note:** While these PDFs explicitly provide the pipelines and structured guidelines to generate data for **Subtask 3 (ST3)** and **Subtask 4 (ST4)**, the exact same procedures can be trivially modified to generate data for **Subtask 1 (ST1)** and **Subtask 2 (ST2)** respectively. The primary difference is simply generating the syllogisms for a single language instead of multilingual generation.

## Setup Requirements

Before running the scripts, please set up your Python environment:

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution

After setup, refer to the individual scripts in `data_generation/` and `training_scripts/` for run instructions specific to each subtask.
