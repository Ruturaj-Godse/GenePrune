'''Utility functions required for the main pruning logic.'''

from functools import reduce
from typing import Union

import torch
import torch.nn as nn

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_module_by_name(module: Union[torch.Tensor, nn.Module],
                       access_string: str):
    """
    Retrieves a sub-module from a PyTorch module hierarchy using a dot-separated access string.

    Parameters:
        module (Union[torch.Tensor, nn.Module]): The root module from which to retrieve the sub-module.
        access_string (str): A string that specifies the path to the sub-module using dot notation. 
                             For example, 'layer1.0.conv1' refers to the 'conv1' module within the 
                             first sub-module of 'layer1'.

    Returns:
        nn.Module: The sub-module at the specified path within the module hierarchy. If the path is 
                   incorrect or the specified module does not exist, it raises an AttributeError.

    This function facilitates dynamic access to any part of a complex model architecture without 
    hard-coding the access to sub-modules, thus enhancing the flexibility and maintainability of 
    code that needs to interact with specific parts of a model.
    """

    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


def get_module_names(module, parent_name, module_list):
    """
    Recursively collects the names of all modules within a given PyTorch module hierarchy.

    Parameters:
        module (nn.Module): The root module from which to start collecting module names.
        parent_name (str): The dot-separated prefix representing the hierarchical path to the current module.
                           This should be an empty string for the root module.
        module_list (list): A list that accumulates the names of all modules. This should initially be an empty list 
                            which will be populated by the function.

    This function traverses the module hierarchy starting from the given module and appends each module's 
    name to the module_list, formatted as a dot-separated string that represents its path within the model 
    (e.g., 'module.submodule.subsubmodule').

    Example usage:
        model = torchvision.models.resnet18(pretrained=True)
        modules = []
        get_module_names(model, '', modules)
        print(modules)  # Output will include names like 'conv1', 'layer1.0.conv1', etc.

    This function is particularly useful for dynamic inspection of models, allowing programmers to list or access 
    components of a model without prior knowledge of its architecture. This can be crucial for tasks such as 
    automated model modification, visualization, or selective parameter freezing / masking.
    """

    module_name = parent_name
    children = [child for child in module.named_children()]
    if children:
        for name, child in children:
            child_name = module_name + ('.' if module_name else '') + name
            get_module_names(child, child_name, module_list)
    else:
        module_list.append(module_name)


def get_bleu_score(model, tokenizer, test_dataloader, test_dataset_df):
    """
    Calculate the BLEU score for a given model and dataset.
    Args:
        model (torch.nn.Module): The model to generate text.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to decode generated text.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        test_dataset_df (pandas.DataFrame): DataFrame containing the test dataset with reference texts.
    Returns:
        float: The average BLEU score of the generated text compared to the reference text.
    """
    
    
    generation = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids = batch['input_ids'].squeeze(1)
            input_ids = input_ids.to(device)
            attention_mask = batch["attention_mask"].squeeze(1)
            attention_mask = attention_mask.to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=128
            )
            generated_code = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generation += generated_code

    test_dataset_df['generation'] = None
    test_dataset_df['generation'] = generation

    bleu = []

    # Define smoothing function
    smooth_func = SmoothingFunction().method1

    for i in range(len(test_dataset_df)):
        reference = str(test_dataset_df.iloc[i]['code'])
        generation = str(test_dataset_df.iloc[i]['generation'])
        code_tokenized = [reference.split()]
        generation_tokenized = generation.split()

        # Calculate BLEU score with smoothing
        bleu_score = sentence_bleu(code_tokenized, generation_tokenized,
                                   weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_func)

        bleu.append(bleu_score)

    # Calculate the average BLEU score
    average_bleu = sum(bleu) / (len(bleu))

    return average_bleu


def fine_tune(model, train_loader, val_loader, epochs, learning_rate=1e-4):
    """
    Fine-tunes a given model using the provided training and validation data loaders.
    Args:
        model (torch.nn.Module): The model to be fine-tuned.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        epochs (int): Number of epochs to train the model.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-4.
    Returns:
        None
    The function performs the following steps:
        1. Sets up the device (GPU if available, otherwise CPU).
        2. Moves the model to the selected device.
        3. Sets up the AdamW optimizer with the specified learning rate.
        4. Iterates over the specified number of epochs:
            - Trains the model on the training dataset, printing the average and current loss every 10 batches.
            - Evaluates the model on the validation dataset, printing the average and current loss every 10 batches.
        5. Clears the GPU cache after each epoch.
    """
    

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)

    # Set up the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1}/{epochs}:")
        # Training
        total_loss = 0
        batch_num = 1
        for batch in train_loader:
            # Load input_ids, attention_mask and labels from batch
            input_ids, attention_mask, labels = batch
            # Resize the tensor dimension for adapting (batch_size, sequence_length)
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            labels = labels.squeeze(1)
            # Pass the data into the device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # Initialize the gradient to zero,and the derivative of loss with respect to weight becomes zero
            optimizer.zero_grad()
            # Call forward function: pass the data into the model,and propagate forward to calculate the predicted value
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # Calculate loss
            loss = outputs.loss
            # Backpropagation for gradient
            loss.backward()
            # Update all parameters of the model
            optimizer.step()

            total_loss += loss.item()

            if (batch_num % 10 == 0) or (batch_num == len(train_loader)):
                print(f'Train: batch: {batch_num}/{len(train_loader)}, Average loss:{total_loss/batch_num}, Current loss:{loss.item()}')

            batch_num += 1
        train_loss = total_loss / len(train_loader)
        print(f"Train Loss: {train_loss}")

        # Validation
        model.eval()
        total_loss = 0
        batch_num = 1
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch

                input_ids = input_ids.squeeze(1)
                attention_mask = attention_mask.squeeze(1)
                labels = labels.squeeze(1)

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                if batch_num % 10 == 0 :
                    print(f'Valid: batch: {batch_num}/{len(val_loader)}, Average loss:{total_loss/batch_num}, Current Loss:{loss.item()}')
                batch_num += 1
            valid_loss = total_loss / len(val_loader)
            print(f"Valid Loss: {valid_loss}")

            # Clear the GPU cache
            torch.cuda.empty_cache()