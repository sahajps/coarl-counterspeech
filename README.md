# coarl-counterspeech

## Getting Started
1. Make sure you have `git`, `python(>=3.8, <3.10)`, [`poetry`](https://python-poetry.org/docs/#installation) installed. Preferably within a virtual environment.

      ```
      pip install poetry
      ```

2. Install dependencies
      ```shell
      cd coarl-counterspeech
      poetry install
      git init
      git add .
      git commit -m "add: initial commit."
      ```


---

## Data Setup

To set up the dataset and prepare the environment for preprocessing and other pipelines, please follow the steps below.

### Prerequisites
- Ensure you have a Huggingface account and access to the **IntentConanV2** dataset [Aswini123/IntentCONANv2](https://huggingface.co/datasets/Aswini123/IntentCONANv2).
- Huggingface's `datasets` library and the required dependencies must be installed. You can install them with the following command:

```bash
pip install datasets
```

### Steps

1. **Run the setup script:**

   The setup script will:
   - Prompt you to log in to your Huggingface account.
   - Download the **IntentConanV2** dataset from Huggingface.
   - Execute data preprocessing and other prompt-related pipelines necessary for the project.

   To run the setup script, use the following command:

   ```bash
   bash setup.sh
   ```

2. **Login to Huggingface:**

   Upon running the script, you will be prompted to log in to your Huggingface account. Make sure you have the necessary access to the dataset - https://huggingface.co/datasets/Aswini123/IntentCONANv2

### Example:

```bash
huggingface-cli login
```

3. **Dataset Download and Preprocessing:**

   After successful login, the script will automatically download the **IntentConanV2** dataset and run the required data preprocessing.

---

### Training Pipelines

Once the `project/data` folder is populated with the necessary data, you can train the following pipelines by running the respective scripts:

- **Multitask Pipeline**: Run the multitask training pipeline by executing:

  ```bash
  bash multitask.sh
  ```

- **PEFT (Parameter-Efficient Fine-Tuning) Pipeline**: Run the PEFT training pipeline by executing:

  ```bash
  bash peft.sh
  ```

- **PPO (Proximal Policy Optimization) Pipeline**: Run the PPO training pipeline by executing:

  ```bash
  bash ppo.sh
  ```

Make sure that the `project/data` folder is fully populated before running any of these scripts.

---


## Directory Structure

| File                                      | Description                                                                  |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| **project**                               | Main directory containing all the code            |
| **project/creds**                         | Directory containing all API access credentials ( project-debator / open-ai / aws)|
| **project/runs**                              | Directory to keep track of all model runs (train / eval). For each run, we store the best_model, classfication args, eval results, metrics, etc.  |
| **project/utils**                             | Program containing utility functions              |
| **project/constants**                         | Program for accessing costant variables, shared variables or default configs   |
| **CHANGELOG.md**                          | Track changes in the code, datasets, etc.                                    |
| **LICENSE**                               | Need to update  |
| **pyproject.toml**                        | Track dependencies here. Also, this means you would be using poetry.         |
| **README.md**                             | This must ring a bell.                                                       |


## Citation
If you find this repository useful in your research, please cite the following paper:

```
@misc{hengle2024intentconditioned,
      title={Intent-conditioned and Non-toxic Counterspeech Generation using Multi-Task Instruction Tuning with RLAIF}, 
      author={Amey Hengle and Aswini Kumar and Sahajpreet Singh and Anil Bandhakavi and Md Shad Akhtar and Tanmoy Chakroborty},
      year={2024},
      eprint={2403.10088},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
