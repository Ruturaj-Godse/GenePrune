# GenePrune

GenePrune is a repository focused on fine-tuning and pruning the CodeT5 model for specific tasks using genetic algorithm-based techniques. It includes scripts for training, pruning, and evaluating the sparsity of neural network models, particularly targeting the reduction of model parameters while maintaining performance.

# Running Experiments with pruning.py

This guide provides instructions on how to run experiments using the pruning.py file. The pruning.py script is designed to prune and fine-tune the CodeT5 model.

## Prerequisites

Before running the experiments, ensure you have the following installed:

- Python 3.6 or higher
- PyTorch 2.5.0
- Other dependencies listed in `requirements.txt`

## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/Ruturaj-Godse/GenePrune.git
    cd GenePrune
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

### Pruning CodeT5

To prune the CodeT5 model, use the prune_codeT5 function in the pruning.py file. The function sets up the model, tokenizer, and datasets, and then prunes the model layers.

1. **Prune a particular layer:**

    ```sh
    python pruning.py --layer <layer_name> --exp_name <experiment_name>
    ```
    For example:
   ```sh
    python pruning.py --layer encoder.block.8.layer.1.DenseReluDense.wo --exp_name exp1
    ```

2. **Prune all layers:**
    
    ```sh
    python pruning.py --save_results --exp_name <experiment_name>
    ```
    

### Fine-Tuning CodeT5

To fine-tune the CodeT5 model, use the fine_tune_codeT5 function in the pruning.py file. The function sets up the model, tokenizer, and datasets, and then fine-tunes the model.

1. **Run the fine-tuning script:**

    ```sh
    python pruning.py --finetune --exp_name <experiment_name>
    ```


