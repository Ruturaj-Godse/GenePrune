# GenePrune

# Running Experiments with pruning.py

This guide provides instructions on how to run experiments using the pruning.py file. The pruning.py script is designed to prune and fine-tune the CodeT5 model.

## Prerequisites

Before running the experiments, ensure you have the following installed:

- Python 3.6 or higher
- PyTorch 2.5.0
- Transformers library from Hugging Face
- Other dependencies listed in `requirements.txt`

## Setup

1. **Clone the repository:**

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Prepare the datasets:**

    Ensure that the datasets are available in the data/pythonapi directory:
    - `api-mined_train.csv`
    - `api-mined_valid.csv`
    - `api-mined_test.csv`

## Running the Experiments

### Fine-Tuning CodeT5

To fine-tune the CodeT5 model, use the fine_tune_codeT5 function in the pruning.py file. The function sets up the model, tokenizer, and datasets, and then fine-tunes the model.

1. **Run the fine-tuning script:**

    ```sh
    python pruning.py --task fine_tune --model_path final_model --exp_name <experiment_name>
    ```

### Pruning CodeT5

To prune the CodeT5 model, use the prune_codeT5 function in the pruning.py file. The function sets up the model, tokenizer, and datasets, and then prunes the model layers.

1. **Run the pruning script:**

    ```sh
    python pruning.py --task prune --model_path final_model --layer_name <layer_name> --exp_name <experiment_name>
    ```

