'''Main logic for automated pruning of neural networks using genetic algorithm.'''

import argparse
import copy
from heapq import heapify, heappush, heappop
import pickle
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.profiler
from tqdm import tqdm

from utils import get_module_by_name, get_module_names, get_bleu_score, fine_tune

from transformers import RobertaTokenizer, T5ForConditionalGeneration
from Datasets import Conala_Dataset, Api_Dataset, Mbpp_Dataset
import os, torch

from Datasets import Test_Dataset
from torch.utils.data import DataLoader

import pandas as pd

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")


args = None             # command line arguments
solution_mask = None    # pruning mask to be shared with the custom pruning function 

MBPP_Val_DF = pd.read_csv('data/mbpp/mbpp_valid.csv')
MBPP_Test_DF = pd.read_csv('data/mbpp/mbpp_test.csv')

START_TIME = time.time()

class CustomPruningMethod(prune.BasePruningMethod):
    """
    A custom pruning method that extends PyTorch's BasePruningMethod to implement
    an unstructured pruning technique using a solution mask provided.

    Attributes:
        PRUNING_TYPE (str): Defines the type of pruning as 'unstructured'. This means
            the pruning is not restricted to any particular structure like channels or
            layers, but can occur at individual weight levels across the model.

    Methods:
        compute_mask(t, default_mask):
            Computes a new mask for the tensor 't' using a globally defined 'solution_mask'
            that specifies which elements of the tensor to prune.
    """

    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        """
        Computes and applies a custom pruning mask to the given tensor.
        
        Parameters:
            t (torch.Tensor): The tensor to be pruned.
            default_mask (torch.Tensor): The default binary mask provided by the pruning method.
        
        Returns:
            torch.Tensor: A new mask tensor that has been customized based on the global 'solution_mask'.
        """

        global solution_mask
        if len(solution_mask) != t.numel():
            mask = torch.ones(t.shape)
        else:
            mask = torch.reshape(solution_mask, t.shape)
        mask = mask.to('cuda')
        return mask
    
    
def custom_unstructured(module, name):
    """
    Applies the CustomPruningMethod to a specific module of a neural network. 
    This function allows for the unstructured pruning of the module's specified 
    parameter (typically weights) using a globally defined pruning mask.

    Parameters:
        module (torch.nn.Module): The module from a neural network whose parameter 
                                  is to be pruned.
        name (str): The name of the parameter within the module to prune, e.g., 'weight'.

    Returns:
        torch.nn.Module: The same module with the specified parameter now subjected to 
                         the custom pruning process. This allows for in-place modification
                         and reusability of the module in further operations or training.
    """
    CustomPruningMethod.apply(module, name)
    return module


def formula(sparsity, accuracy):
    """
    Computes the objective function value for a given sparsity and accuracy of a pruned neural network. 
    This function calculates a weighted sum of sparsity and accuracy to evaluate the trade-off 
    between model complexity (as measured by sparsity) and model performance (as measured by accuracy).

    Parameters:
        sparsity (float): The sparsity of the model, representing the proportion of the model's 
                          weights that have been pruned, typically a value between 0 and 1.
        accuracy (float): The accuracy of the model on a validation or test dataset, typically 
                          a value between 0 and 1.

    Returns:
        float: The computed value of the objective function, representing the trade-off between 
               sparsity and accuracy.

    The function uses a fixed weight `alpha` of 0.8 to prioritize sparsity, but this can be adjusted 
    depending on specific requirements or preferences for the balance between sparsity and accuracy.
    """
    alpha = 0.02
    return alpha * sparsity + (1 - alpha) * accuracy


def objective_function_code(model, tokenizer, layer_name, solution, test_loader, test_dataset_df, accuracy_lower_limit=0, ret=0):

    global solution_mask
    temp_model = copy.deepcopy(model)
    solution_mask = torch.tensor(solution)
    layer = get_module_by_name(temp_model, layer_name)
    custom_unstructured(layer, name='weight')
    
    acc = get_bleu_score(
        temp_model, 
        tokenizer, 
        test_loader, 
        test_dataset_df,
    )

    spar = (solution.size - np.count_nonzero(solution))/solution.size

    if ret == 1:
        return (spar, acc)
    
    mf = 1
    if acc < accuracy_lower_limit:
        mf = 0.1
    return formula(spar, acc) * mf


class GeneticPruning:
    """
    Implements genetic algorithm for pruning neural networks. This class handles the creation
    of an initial population of pruning masks, the evaluation of these masks based on network performance,
    and the evolution of the population over generations to optimize network sparsity while maintaining
    or improving accuracy. It provides methods to prune one or more layers of the model using the above approach.

    Attributes:
        model (torch.nn.Module): The neural network model to be pruned.
        model_name (str): Name of the model, automatically derived from the model's class.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        optimizer (torch.optim.Optimizer): Optimizer used for training the pruned model.
        criterion (torch.nn.Module): Loss function used during training.
        post_prune_epochs (int): Number of epochs to train the model after each pruning iteration.
        post_prune_epochs_per_layer (int): Number of epochs to to train the model after pruning each layer.
        device (torch.device): Device on which to perform computations (e.g., 'cuda' or 'cpu').
        module_names ([str]): List of all module names in the model.
        layer_name (str): Name of the current layer being pruned.
        layer_solutions (dict): Dictionary storing solutions (best prune masks) for each layer.
        accuracy_lower_limit (float): Minimum acceptable accuracy for the pruned model.
    """



    def __init__(self, model, train_loader, val_loader, val_test_loader, test_loader, optimizer, criterion, device, post_prune_epochs, post_prune_epochs_per_layer, accuracy_lower_limit, bleu_lower_limit, tokenizer) -> None:
        self.model = model
        self.model_name = model.__class__.__name__
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_test_loader = val_test_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.post_prune_epochs = post_prune_epochs
        self.post_prune_epochs_per_layer = post_prune_epochs_per_layer
        self.device = device
        self.module_names = []
        get_module_names(self.model, '', self.module_names)
        self.layer_name = None
        self.layer_solutions = {}
        self.accuracy_lower_limit = accuracy_lower_limit
        self.blue_lower_limit = bleu_lower_limit
        self.tokenizer = tokenizer
        self.layer_limit = 2
        self.filter_and_sort_module_names()

    def filter_and_sort_module_names(self):
        layer_size_pairs = []
        for layer_name in self.module_names:
            layer = get_module_by_name(self.model, layer_name)
            try:
                layer_size_pairs.append((layer_name, np.prod(layer.weight.size())))
            except:
                pass
        layer_size_pairs.sort(key=lambda x: x[1], reverse=True)
        self.module_names = [name for name, _ in layer_size_pairs]
        # remove embedding module names
        self.module_names = [name for name in self.module_names if 'embed_tokens' not in name]

        self.module_names = [
            'encoder.block.0.layer.0.SelfAttention.q',
            'encoder.block.0.layer.0.SelfAttention.k',
            'encoder.block.0.layer.0.SelfAttention.v',
            'encoder.block.0.layer.0.SelfAttention.o',
            'encoder.block.0.layer.0.SelfAttention.relative_attention_bias',
            'encoder.block.0.layer.0.layer_norm',
            'encoder.block.0.layer.1.DenseReluDense.wi',
            'encoder.block.0.layer.1.DenseReluDense.wo',
            'encoder.block.0.layer.1.layer_norm',
        ]

        # self.module_names = [ 
        #     'shared',
        #     'encoder.block.0.layer.0.SelfAttention.v',
            # 'encoder.block.8.layer.1.DenseReluDense.wo',
            # 'encoder.block.10.layer.0.SelfAttention.k',
            # 'encoder.block.5.layer.1.DenseReluDense.wi',
            # 'encoder.block.10.layer.1.DenseReluDense.wo',
            # 'encoder.block.9.layer.1.DenseReluDense.wi',
            # 'decoder.block.2.layer.1.EncDecAttention.v',
            # 'encoder.block.4.layer.1.DenseReluDense.wo',
            # 'encoder.block.0.layer.0.SelfAttention.o',
            # 'decoder.block.3.layer.1.EncDecAttention.o',
            # 'decoder.block.10.layer.2.DenseReluDense.wo',
            # 'encoder.block.1.layer.0.SelfAttention.k',
            # 'encoder.block.4.layer.1.DenseReluDense.wi',
            # 'decoder.block.0.layer.1.EncDecAttention.k',
            # 'encoder.block.10.layer.0.SelfAttention.q',
            # 'decoder.block.0.layer.2.DenseReluDense.wo',
            # 'encoder.block.4.layer.0.SelfAttention.k',
            # 'decoder.block.10.layer.2.DenseReluDense.wi',
            # 'decoder.block.0.layer.1.EncDecAttention.q',
            # 'encoder.block.6.layer.0.SelfAttention.k',
            # 'encoder.block.9.layer.1.DenseReluDense.wo',
            # 'decoder.block.11.layer.1.EncDecAttention.v',
            # 'encoder.block.9.layer.1.layer_norm',
            # 'decoder.block.7.layer.2.DenseReluDense.wo',
            # 'encoder.block.4.layer.0.SelfAttention.v',
            # 'decoder.block.2.layer.1.EncDecAttention.q',
            # 'decoder.block.4.layer.2.DenseReluDense.wi',
            # 'encoder.block.11.layer.1.DenseReluDense.wi',
            # 'encoder.block.7.layer.1.DenseReluDense.wo',
            # 'decoder.block.2.layer.1.EncDecAttention.k',
            # 'encoder.block.1.layer.1.DenseReluDense.wo',
            # 'encoder.block.3.layer.1.DenseReluDense.wi',
            # 'encoder.block.5.layer.1.layer_norm',
            # 'encoder.block.1.layer.0.SelfAttention.v',
            # 'decoder.block.5.layer.1.EncDecAttention.k',
            # 'encoder.block.0.layer.0.SelfAttention.q',
            # 'decoder.block.7.layer.2.DenseReluDense.wi',
            # 'decoder.block.3.layer.1.EncDecAttention.k',
            # 'encoder.block.1.layer.0.SelfAttention.q',
            # 'encoder.block.2.layer.1.DenseReluDense.wi',
            # 'decoder.block.11.layer.1.EncDecAttention.q',
            # 'decoder.block.0.layer.1.EncDecAttention.o',
            # 'decoder.block.1.layer.2.DenseReluDense.wo',
            # 'encoder.block.6.layer.1.DenseReluDense.wo',
            # 'encoder.block.0.layer.1.DenseReluDense.wo',
            # 'decoder.block.10.layer.1.EncDecAttention.o',
            # 'decoder.block.11.layer.2.DenseReluDense.wo',
            # 'encoder.block.2.layer.1.DenseReluDense.wo',
            # 'encoder.block.1.layer.1.DenseReluDense.wi',
            # 'decoder.block.3.layer.2.DenseReluDense.wo',
            # 'decoder.block.0.layer.1.EncDecAttention.v',
            # 'encoder.block.4.layer.0.SelfAttention.q',
            # 'decoder.block.0.layer.2.DenseReluDense.wi',
            # 'decoder.block.5.layer.2.DenseReluDense.wi',
            # 'decoder.block.3.layer.1.EncDecAttention.q',
            # 'decoder.block.3.layer.2.DenseReluDense.wi',
            # 'encoder.block.7.layer.1.DenseReluDense.wi',
            # 'decoder.block.2.layer.2.DenseReluDense.wo',
            # 'encoder.block.6.layer.1.DenseReluDense.wi',
            # 'encoder.block.0.layer.0.SelfAttention.k',
            # 'decoder.block.6.layer.2.DenseReluDense.wo',
            # 'encoder.block.10.layer.1.DenseReluDense.wi',
            # 'encoder.block.11.layer.1.DenseReluDense.wo',
            # 'encoder.block.0.layer.1.DenseReluDense.wi',
            # 'encoder.block.4.layer.0.SelfAttention.o',
            # 'decoder.block.8.layer.2.DenseReluDense.wo',
            # 'decoder.block.9.layer.2.DenseReluDense.wo',
            # 'decoder.block.4.layer.2.DenseReluDense.wo',
            # 'decoder.block.2.layer.2.DenseReluDense.wi',
            # 'decoder.block.11.layer.2.DenseReluDense.wi',
            # 'decoder.block.5.layer.2.DenseReluDense.wo',
            # 'decoder.block.9.layer.2.DenseReluDense.wi',
            # 'decoder.block.6.layer.2.DenseReluDense.wi',
            # 'decoder.block.2.layer.1.EncDecAttention.o',
            # 'encoder.block.8.layer.1.DenseReluDense.wi',
            # 'encoder.block.5.layer.1.DenseReluDense.wo',
            # 'encoder.block.1.layer.0.SelfAttention.o',
            # 'decoder.block.1.layer.2.DenseReluDense.wi',
            # 'decoder.block.3.layer.1.EncDecAttention.v',
            # 'encoder.block.3.layer.1.DenseReluDense.wo',
        # ]

    
    def objective_function_wrapper(self, solution, ret=0):
        """
        Wrapper method for the objective_function to evaluate the efficacy of a pruning solution based 
        on model sparsity and accuracy using the validation dataset. This method allows for easy integration 
        of the objective function within the genetic algorithm workflow by automatically passing the relevant 
        model, layer, and data loader.

        Parameters:
            solution (np.array): The pruning mask to be applied, typically a binary array where 1 indicates 
                                a weight is kept and 0 indicates it is pruned.
            ret (int, optional): A flag to determine the type of return value:
                                - If 0, returns the computed objective function value (default).
                                - If 1, returns a tuple containing sparsity and accuracy of the model 
                                after applying the pruning solution.

        Returns:
            float or tuple: Depending on the `ret` value, returns either the objective function value 
                            or a tuple (sparsity, accuracy) reflecting the performance metrics of 
                            the pruned model.
        """
        # return objective_function_code(self.model, self.tokenizer, self.layer_name, solution, self.test_loader, MBPP_Test_DF, self.blue_lower_limit, ret)
        return objective_function_code(self.model, self.tokenizer, self.layer_name, solution, self.val_test_loader, MBPP_Val_DF, self.blue_lower_limit, ret)
    

    def performance_wrapper(self, solution, ret=0):
        """
        Wrapper method for the objective_function to evaluate the efficacy of a pruning solution based 
        on model sparsity and accuracy using the test dataset (instead of validation dataset). This method 
        allows for easy integration of the objective function within the genetic algorithm workflow by automatically 
        passing the relevant model, layer, and data loader.

        Parameters:
            solution (np.array): The pruning mask to be applied, typically a binary array where 1 indicates 
                                a weight is kept and 0 indicates it is pruned.
            ret (int, optional): A flag to determine the type of return value:
                                - If 0, returns the computed objective function value (default).
                                - If 1, returns a tuple containing sparsity and accuracy of the model 
                                after applying the pruning solution.

        Returns:
            float or tuple: Depending on the `ret` value, returns either the objective function value 
                            or a tuple (sparsity, accuracy) reflecting the performance metrics of 
                            the pruned model.
        """
        return objective_function_code(self.model, self.tokenizer, self.layer_name, solution, self.test_loader, MBPP_Test_DF, self.blue_lower_limit, ret)
    

    def initial_population(self, pop_size, solution_size, initial_sparsity_ratio):
        """
        Generates an initial population of pruning masks for the genetic algorithm. Each individual in 
        the population represents a potential solution (pruning mask) for the model. This method initializes 
        the population with random binary arrays where the probability of each element being zero (pruned) 
        is determined by the initial sparsity ratio.

        Parameters:
            pop_size (int): The size of the population, i.e., the number of individual solutions to generate.
            solution_size (int): The size of each solution, which should match the number of parameters 
                                in the layer of the model that is being targeted for pruning.
            initial_sparsity_ratio (float): The proportion of weights to initially set as pruned (0) in each 
                                            solution. The rest will be set to keep (1).

        Returns:
            list of np.array: A list containing the initial population of solutions, where each solution 
                            is a numpy array of binary values (0s and 1s).
        """
        population = [np.random.choice([0, 1], size=(solution_size,), p=[initial_sparsity_ratio, 1-initial_sparsity_ratio]) for _ in range(pop_size)]
        return population
    

    def crossover(self, parent1, parent2, crossover_rate=0.9):
        """
        Performs the crossover operation in genetic algorithm to generate new offspring (solutions) from two parent solutions.
        This method uses a single-point crossover approach where a point on the parent solution arrays is chosen at random,
        and the tails beyond that point are swapped between the two parents to create two new children.

        Parameters:
            parent1 (np.array): The first parent solution array.
            parent2 (np.array): The second parent solution array.
            crossover_rate (float, optional): The probability of the crossover operation occurring between two parents.
                                            If a random draw falls below this rate, crossover happens; otherwise,
                                            the parents are returned without modification. Defaults to 0.9.

        Returns:
            tuple of np.array: A tuple containing two new solutions (children), each being a numpy array. These children are
                            either a result of the crossover (if it occurs) or direct copies of the original parents (if not).
        """
        if random.random() < crossover_rate:
            point = random.randint(1, len(parent1)-1)  # Crossover point
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
        

    def mutation(self, solution, mutation_rate=0.01):
        """
        Applies mutation to a given solution, altering its elements to introduce variability and prevent premature convergence.
        This method iterates through each element of the solution array, and with a probability defined by the mutation rate,
        sets the element to 0 (representing a pruned weight). This stochastic process helps to explore new areas of the solution
        space that may not be reachable through crossover alone.

        Parameters:
            solution (np.array): The solution array to be mutated, typically a binary array where each element indicates
                                whether a corresponding weight is kept (1) or pruned (0).
            mutation_rate (float, optional): The probability of any single weight being mutated (set to 0). Defaults to 0.01.

        Returns:
            np.array: The mutated solution array.

        Mutation is a fundamental genetic operator that provides genetic diversity and enables the genetic algorithm to
        escape local optima by randomly altering solution elements. This method ensures that even well-performing solutions
        are subjected to random changes, thus simulating natural genetic variation and fostering robustness in the solution population.
        """
        for i in range(len(solution)):
            if random.random() < mutation_rate:
                solution[i] = 0
        return solution
    
    
    def load_population(self, population_size, solution_size, initial_sparsity_ratio):
        """
        Loads a population of pruning solutions from a file, or generates a new population if the file does not exist.
        This method attempts to retrieve a previously saved population specific to the model and layer from a pickle file.
        If the file cannot be found or an error occurs during the load process, it generates a new initial population
        using the specified sparsity ratio.

        Parameters:
            population_size (int): The number of individual solutions (population size) to be loaded or generated.
            solution_size (int): The number of parameters in the targeted layer, determining the size of each solution.
            initial_sparsity_ratio (float): The proportion of weights to initially set as pruned (0) in each solution,
                                            used if a new population needs to be generated.

        Returns:
            list of np.array: A population of solutions, where each solution is a numpy array of binary values (0s and 1s).
                            Each array represents a potential pruning mask for the neural network layer.
        """
        try:
            with open(f"./populations/{self.model_name}/{args.exp_name}/population_{population_size}_{self.layer_name}.pkl", 'rb') as fp:
                population = pickle.load(fp)
        except:
            population = self.initial_population(population_size, solution_size, initial_sparsity_ratio)
        return population
    
    
    def genetic_algorithm(self, population_size, solution_size, crossover_rate, mutation_rate, generations, warm_start, initial_sparsity_ratio, sparsity_threshold):
        """
        Executes the genetic algorithm to optimize pruning masks for a neural network based on a defined objective function
        that evaluates sparsity and accuracy. The method handles initialization of the population, selection, crossover,
        mutation, and replacement over a number of generations.

        Parameters:
            population_size (int): The number of solutions in the population.
            solution_size (int): The number of parameters in the targeted layer, determining the size of each solution.
            crossover_rate (float): Probability with which two solutions will undergo crossover.
            mutation_rate (float): Probability with which any single element of a solution may be mutated.
            generations (int): Number of iterations the genetic algorithm should run.
            warm_start (bool): If True, the population is loaded from a previously saved state; otherwise, it is initialized anew.
            initial_sparsity_ratio (float): The proportion of weights initially set as pruned when creating a new population.
            sparsity_threshold (float): The sparsity level at which the algorithm will stop if achieved by any solution.

        Returns:
            tuple: A tuple containing the best score achieved and the corresponding best solution.

        This method advances through multiple generations, each time performing selection based on fitness, 
        breeding new solutions through crossover and mutation, and inserting them into the population using 
        a heap-based selection strategy to keep only the best solutions. It tracks and reports performance metrics 
        across generations, including average and best scores, and validation and test accuracies.
        The method also supports saving the state of the population and the best solution per generation if 
        required by the implementation settings, which allows for resuming the process or auditing the results later.
        """
        global args

        if warm_start:
            population = self.load_population(population_size, solution_size, initial_sparsity_ratio)
        else:
            population = self.initial_population(population_size, solution_size, initial_sparsity_ratio)
        population_heap = [(self.objective_function_wrapper(sol), idx, sol) for idx, sol in enumerate(population)]
        available_indices = set([len(population_heap), len(population_heap)+1])
        heapify(population_heap)
        for gen in range(generations):
            for i in tqdm(range(len(population_heap))):
                _, _, x = random.choice(population_heap)
                _, _, y = random.choice(population_heap)
                c1, c2 = self.crossover(x, y, crossover_rate)
                c1 = self.mutation(c1, mutation_rate)
                c2 = self.mutation(c2, mutation_rate)
                idx1 = available_indices.pop()
                idx2 = available_indices.pop()
                heappush(population_heap, (self.objective_function_wrapper(c1), idx1, c1))
                heappush(population_heap, (self.objective_function_wrapper(c2), idx2, c2))
                _, idx1, _ = heappop(population_heap)
                _, idx2, _ = heappop(population_heap)
                available_indices.add(idx1)
                available_indices.add(idx2)

            best_score, _, best_sol = max(population_heap, key= lambda x : x[0])
            val_perf = self.objective_function_wrapper(best_sol, ret=1)
            test_perf = self.performance_wrapper(best_sol, ret=1)
            avg_score = sum(val for val, _, _ in population_heap)/len(population_heap)
            print(f"Generation {gen + 1}: Best Score = {best_score:.4f} | Best Sparsity {test_perf[0]:.4f} | Val Accuracy: {val_perf[1]:.4f} | Test Accuracy: {test_perf[1]:.4f} | Avg Score = {avg_score:.4f}")

            population = [sol for _, _, sol in population_heap]
            if args.save_results:
                directory = f"./populations/{self.model_name}/{args.exp_name}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                with open(f"./populations/{self.model_name}/{args.exp_name}/population_{population_size}_{self.layer_name}.pkl", 'wb') as fp:
                    pickle.dump(population, fp)

            if args.save_results:
                directory = f"./solutions/{self.model_name}/{args.exp_name}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                with open(f"./solutions/{self.model_name}/{args.exp_name}/best_solution_{self.layer_name}.pkl", 'wb') as fp:
                    pickle.dump(best_sol, fp)

            if val_perf[0] >= sparsity_threshold:
                break

        best_score, _, best_sol = max(population_heap, key= lambda x : x[0])
        return best_score, best_sol

    def mock_genetic_algorithm(self, population_size, solution_size, crossover_rate, mutation_rate, generations, warm_start, initial_sparsity_ratio, sparsity_threshold):
        """
        Mock genetic algorithm that generates random pruning masks instead of evolving them.
        This function mimics the interface of the genetic algorithm but uses random masks.

        Parameters:
            population_size (int): The number of solutions in the population.
            solution_size (int): The number of parameters in the targeted layer, determining the size of each solution.
            crossover_rate (float): Not used in this mock function.
            mutation_rate (float): Not used in this mock function.
            generations (int): Not used in this mock function.
            warm_start (bool): Not used in this mock function.
            initial_sparsity_ratio (float): The proportion of weights initially set as pruned when creating a new population.
            sparsity_threshold (float): The sparsity level at which the algorithm will stop if achieved by any solution.

        Returns:
            tuple: A tuple containing the best score achieved and the corresponding best solution.
        """
        
        best_solution = np.random.choice([0, 1], size=(solution_size,), p=[initial_sparsity_ratio, 1-initial_sparsity_ratio])
        best_score = self.objective_function_wrapper(best_solution)

        return best_score, best_solution
    
    def prune_one_layer(self, layer_name, config):
        """
        Prunes a specified layer of a neural network using the genetic algorithm configured through `config`. This method 
        orchestrates the pruning process by determining the best pruning solution for the layer, applying this solution, 
        retraining the model on the pruned layer, and finally removing the pruned weights permanently.

        Parameters:
            layer_name (str): The name of the layer to prune.
            config (dict): Configuration parameters for the genetic algorithm including population size, solution size,
                        crossover rate, mutation rate, number of generations, and other relevant settings.

        Returns:
            None. Outputs the best solution's score and performance directly and updates the model in-place.

        This method integrates several steps:
        - It runs the genetic algorithm to find the optimal pruning solution for the specified layer.
        - Applies the best pruning solution to the layer.
        - Retrains the model for a specified number of epochs to adjust to the changes introduced by pruning.
        - Removes the pruned weights permanently from the layer.
        - Evaluates and prints the test accuracy after pruning.
        - Checks and logs the sparsity achieved in the pruned layer.
        """
        global solution_mask

        self.layer_name = layer_name
        layer = get_module_by_name(self.model, self.layer_name)
        
        best_score, best_solution = self.genetic_algorithm(**config)
        print("Best Solution Score:", best_score)
        print("Best Solution Perf:", self.objective_function_wrapper(best_solution, ret=1))

        self.layer_solutions[self.layer_name] = best_solution

        solution_mask = torch.tensor(best_solution)
        layer = get_module_by_name(self.model, self.layer_name)
        custom_unstructured(layer, name='weight')

        fine_tune(self.model, self.train_loader, self.val_loader, epochs=1, learning_rate=1e-6)

        prune.remove(layer, 'weight')

        print("Test accuracy: ", get_bleu_score(self.model, self.tokenizer, self.test_loader, MBPP_Test_DF))

        self.check_sparsity()

    
    def prune_all_layers(self, config):
        """
        Prunes all the layers of a neural network that have trainable weights using a genetic algorithm. This method
        sequentially processes each layer, applies the genetic algorithm to find the optimal pruning solution, retrains
        the model to adapt to the pruned layer, and iteratively updates the entire model's structure.

        Parameters:
            config (dict): Configuration parameters for the genetic algorithm which include settings such as population size,
                        crossover rate, mutation rate, and other necessary parameters to execute the genetic pruning.

        Returns:
            None. This method updates the model in-place and prints out the performance metrics after pruning each layer and
            the entire model.

        The method goes through each layer, checks if it has trainable weights, and if so, proceeds to prune it based on the
        genetic algorithm. After pruning each layer, the model is retrained to ensure it adapts well to the changes. The final
        accuracy of the model on the test set is printed, and if configured, the pruned model's state is saved.

        During the process, it also keeps track of the test accuracies after each layer's pruning and after final retraining,
        providing insights into the model's performance progression as layers are pruned and retrained.
        """
        global solution_mask, args

        test_accuracies = []
        for idx, layer_name in enumerate(self.module_names):
            try:
                self.layer_name = layer_name
                layer = get_module_by_name(self.model, self.layer_name)

                print()
                print(f"Layer number: {idx+1}")
                print(f"Layer name: {layer_name}")

                try:
                    layer.weight
                except:
                    print("No attribute weight!")
                    continue

                print(f"Layer shape: {layer.weight.size()}")
                print(f"Layer size: {np.prod(layer.weight.size())}")
                
                config["solution_size"] = np.prod(layer.weight.size())

                best_score, best_solution = self.genetic_algorithm(**config)
                # best_score, best_solution = self.mock_genetic_algorithm(**config)

                print("Best Solution Score:", best_score)
                print("Best Solution Perf:", self.objective_function_wrapper(best_solution, ret=1))

                self.layer_solutions[self.layer_name] = best_solution

                for prev_layer_name, prev_solution in list(self.layer_solutions.items()):
                    solution_mask = torch.tensor(prev_solution)
                    prev_layer = get_module_by_name(self.model, prev_layer_name)
                    custom_unstructured(prev_layer, name='weight')


                if (idx+1)%5 == 0:
                    print()
                    fine_tune(self.model, self.train_loader, self.val_loader, epochs=2, learning_rate=1e-6)

                for prev_layer_name, prev_solution in list(self.layer_solutions.items()):
                    prev_layer = get_module_by_name(self.model, prev_layer_name)
                    prune.remove(prev_layer, 'weight')

                print("Test accuracy: ", get_bleu_score(self.model, self.tokenizer, self.test_loader, MBPP_Test_DF))

                test_accuracies.append(get_bleu_score(self.model, self.tokenizer, self.test_loader, MBPP_Test_DF))
                print(f"Test accuracy history:", test_accuracies)
                self.check_sparsity()
                if args.save_results:
                    directory = f"./sparse_models/{self.model_name}/{args.exp_name}"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    torch.save(self.model.state_dict(), f'./sparse_models/{self.model_name}/{args.exp_name}/sparse_weights.pth')

                print("Time taken: ", time.time()-START_TIME, "sec")
            except Exception as e:
                print(f"Error in layer {layer_name}: {e}")
                continue

        # for prev_layer_name, prev_solution in self.layer_solutions.items():
        #     solution_mask = torch.tensor(prev_solution)
        #     prev_layer = get_module_by_name(self.model, prev_layer_name)
        #     custom_unstructured(prev_layer, name='weight')

        # print()
        # fine_tune(self.model, self.train_loader, self.val_loader, epochs=3, learning_rate=1e-6)

        # for prev_layer_name, prev_solution in self.layer_solutions.items():
        #     prev_layer = get_module_by_name(self.model, prev_layer_name)
        #     prune.remove(prev_layer, 'weight')

        # print("Test accuracy: ", get_bleu_score(self.model, self.tokenizer, self.test_loader, MBPP_Test_DF))

        # test_accuracies.append(get_bleu_score(self.model, self.tokenizer, self.test_loader, MBPP_Test_DF))
        # print(f"Test accuracy history:", test_accuracies)
        # self.check_sparsity()
        # if args.save_results:
        #     directory = f"./sparse_models/{self.model_name}/{args.exp_name}"
        #     if not os.path.exists(directory):
        #         os.makedirs(directory)
        #     torch.save(self.model.state_dict(), f'./sparse_models/{self.model_name}/{args.exp_name}/sparse_weights_retrained.pth')

        # print("Time taken: ", time.time()-START_TIME, "sec")


    def prune_all_layers_iteratively(self, iters, sparsity_thresholds, config):
        """
        Iteratively prunes all trainable layers of a neural network multiple times across specified iterations, 
        adjusting the sparsity threshold each iteration based on provided thresholds. This method allows for 
        progressively deeper pruning with the ability to fine-tune the model's response to increased sparsity after 
        each iteration.

        Parameters:
            iters (int): Number of iterations to repeat the entire pruning process.
            sparsity_thresholds (list): List of sparsity thresholds for each iteration; controls how aggressive 
                                        the pruning should be in each iteration.
            config (dict): Configuration parameters for the genetic algorithm, including settings such as population 
                        size, crossover rate, mutation rate, and other necessary parameters to execute the genetic pruning.

        Returns:
            None. This method updates the model in-place and prints out performance metrics after pruning each layer 
            and the entire model in each iteration.

        During each iteration, this method goes through each layer of the model, applies genetic algorithm-based pruning, 
        retrains the model to adapt to the new sparsity level, and evaluates the model's performance. This approach 
        helps in achieving a desired global sparsity level while maintaining model performance as much as possible.
        After each iteration, the method optionally saves the state of the model, facilitating further analysis or deployment.
        """
        global solution_mask, args
        
        for iter in range(iters):
            test_accuracies = []
            config['sparsity_threshold'] = sparsity_thresholds[iter]
            for idx, layer_name in enumerate(self.module_names):
                self.layer_name = layer_name
                layer = get_module_by_name(self.model, self.layer_name)

                print()
                print(f"Layer number: {idx+1}")
                print(f"Layer name: {layer_name}")

                try:
                    layer.weight
                except:
                    print("No attribute weight!")
                    continue

                print(f"Layer shape: {layer.weight.size()}")
                print(f"Layer size: {np.prod(layer.weight.size())}")
                
                config["solution_size"] = np.prod(layer.weight.size())
                best_score, best_solution = self.genetic_algorithm(**config)
                print("Best Solution Score:", best_score)
                print("Best Solution Perf:", self.objective_function_wrapper(best_solution, ret=1))

                self.layer_solutions[self.layer_name] = best_solution

                for prev_layer_name, prev_solution in self.layer_solutions.items():
                    solution_mask = torch.tensor(prev_solution)
                    prev_layer = get_module_by_name(self.model, prev_layer_name)
                    custom_unstructured(prev_layer, name='weight')

                if (idx+1)%5 == 0:
                    fine_tune(self.model, self.train_loader, self.val_loader, epochs=5, learning_rate=1e-6)

                for prev_layer_name, prev_solution in self.layer_solutions.items():
                    prev_layer = get_module_by_name(self.model, prev_layer_name)
                    prune.remove(prev_layer, 'weight')

                print("Test accuracy: ", get_bleu_score(self.model, self.tokenizer, self.test_loader, MBPP_Test_DF))

                test_accuracies.append(get_bleu_score(self.model, self.tokenizer, self.test_loader, MBPP_Test_DF))
                print(f"Test accuracy history:", test_accuracies)
                self.check_sparsity()
                if args.save_results:
                    directory = f"./sparse_models/{self.model_name}/{args.exp_name}"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    torch.save(self.model.state_dict(), f'./sparse_models/{self.model_name}/{args.exp_name}/sparse_weights.pth')

            for prev_layer_name, prev_solution in self.layer_solutions.items():
                solution_mask = torch.tensor(prev_solution)
                prev_layer = get_module_by_name(self.model, prev_layer_name)
                custom_unstructured(prev_layer, name='weight')

            fine_tune(self.model, self.train_loader, self.val_loader, epochs=5, learning_rate=1e-6)

            for prev_layer_name, prev_solution in self.layer_solutions.items():
                prev_layer = get_module_by_name(self.model, prev_layer_name)
                prune.remove(prev_layer, 'weight')

            print("Test accuracy: ", get_bleu_score(self.model, self.tokenizer, self.test_loader, MBPP_Test_DF))
            print(f"Test accuracy history:", test_accuracies)

            self.check_sparsity()
            if args.save_results:
                directory = f"./sparse_models/{self.model_name}/{args.exp_name}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save(self.model.state_dict(), f'./sparse_models/{self.model_name}/{args.exp_name}/sparse_weights_retrained.pth')

            config['warm_start'] = True


    def check_sparsity(self):
        """
        Calculates and prints the sparsity levels for each layer in the model that has been pruned using the 
        solutions stored in `layer_solutions`. This method provides a measure of how many weights in each layer 
        have been set to zero (pruned) as a proportion of the total number of weights in that layer.

        Returns:
            None. Outputs the sparsity levels directly to the console.

        This method iterates through each pruned layer, retrieves the current weights from the model, converts them 
        to a NumPy array, and calculates the proportion of weights that are zero. The sparsity level for each layer 
        is then printed, giving an overview of the model's overall reduction in parameter count due to pruning.
        """
        sparsity_levels = []
        for layer_name in self.layer_solutions:
            layer = get_module_by_name(self.model, layer_name)
            layer_weights = copy.deepcopy(layer.weight)
            layer_weights = layer_weights.to('cpu')
            layer_weights = np.array(layer_weights.detach())
            sparsity = (layer_weights.size - np.count_nonzero(layer_weights))/layer_weights.size
            sparsity_levels.append(sparsity)

        print("Sparsity Levels: ", sparsity_levels)


def prune_codeT5(layer_name):
    
    model_path = 'final_model'
    assert os.path.exists(model_path), "Model path does not exist..."
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Your device is {device}.')

    # Set up model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # train_dataset = Api_Dataset('api-mined_train.csv', text_length=128, code_length=128)
    # valid_dataset = Api_Dataset('api-mined_valid.csv', text_length=128, code_length=128)

    train_dataset = Mbpp_Dataset('mbpp_train.csv', text_length=128, code_length=128)
    valid_dataset = Mbpp_Dataset('mbpp_valid.csv', text_length=128, code_length=128)

    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    val_test_dataset = Test_Dataset(data='data/mbpp/mbpp_valid.csv', task_prefix='Generate code from natural language: (from Mbpp)')
    val_test_loader = DataLoader(dataset=val_test_dataset, batch_size=128, shuffle=False)

    test_dataset = Test_Dataset(data='data/mbpp/mbpp_test.csv', task_prefix='Generate code from natural language: (from Mbpp)')
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    GP = GeneticPruning(model, train_loader, val_loader, val_test_loader, test_loader, None, None, device, \
                        post_prune_epochs=10, post_prune_epochs_per_layer=3, accuracy_lower_limit=0.6, bleu_lower_limit=0.09, tokenizer=tokenizer)
    
    if layer_name:
        layer = get_module_by_name(model, layer_name)
        print(f"Layer name: {layer_name}")
        print(f"Layer shape: {layer.weight.size()}")
        print(f"Layer size: {np.prod(layer.weight.size())}")
        
        config = {
            "population_size": 10,
            "solution_size": np.prod(layer.weight.size()),
            "crossover_rate": 1,
            "mutation_rate": 0.05,
            "generations": 5,
            "warm_start": False,
            "initial_sparsity_ratio": 0.1,
            "sparsity_threshold": 1
        }
        GP.prune_one_layer(layer_name, config)
    else:
        print("Layers to be pruned:", GP.module_names)
        config = {
            "population_size": 10,
            "crossover_rate": 1,
            "mutation_rate": 0.1,
            "generations": 5,
            "warm_start": False,
            "initial_sparsity_ratio": 0.05,
            "sparsity_threshold": 0.4
        }
        GP.prune_all_layers(config)


def fine_tune_codeT5():
    global solution_mask, args

    model_path = 'final_model'
    assert os.path.exists(model_path), "Model path does not exist..."
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Your device is {device}.')
    # Set up model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    checkpoint = torch.load(f'./sparse_models/T5ForConditionalGeneration/{args.exp_name}/sparse_weights_finetuned.pth')
    model.load_state_dict(checkpoint)

    train_dataset = Api_Dataset('data/pythonapi/api-mined_train.csv', text_length=128, code_length=128)
    valid_dataset = Api_Dataset('data/pythonapi/api-mined_valid.csv', text_length=128, code_length=128)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    test_dataset = Test_Dataset(data='data/pythonapi/api-mined_test.csv', task_prefix= 'Generate code from natural language: (from Pythonapi)')
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    module_names = []
    get_module_names(model, '', module_names)

    layer_solutions = {}

    for layer_name in module_names:
        try:
            with open(f"./solutions/T5ForConditionalGeneration/{args.exp_name}/best_solution_{layer_name}.pkl", 'rb') as fp:
                best_solution = pickle.load(fp)
        except:
            continue
        layer_solutions[layer_name] = best_solution
        print(f"Layer {layer_name} mask obtained!")

    print()

    for layer_name, solution in layer_solutions.items():
        solution_mask = torch.tensor(solution)
        layer = get_module_by_name(model, layer_name)
        custom_unstructured(layer, name='weight')
        print(f"Layer {layer_name} masked!")
    print()

    print("Test accuracy before fine-tuning: ", get_bleu_score(model, tokenizer, test_loader, MBPP_Test_DF))

    for iter in range(5):
        print(f"Iteration {iter+1}:")
        fine_tune(model, train_loader, val_loader, epochs=2, learning_rate=1e-6)
        print(f"Test accuracy after {(iter+1)*2}: ", get_bleu_score(model, tokenizer, test_loader, MBPP_Test_DF))

    for layer_name, solution in layer_solutions.items():
        layer = get_module_by_name(model, layer_name)
        prune.remove(layer, 'weight')
        print(f"Layer {layer_name} unmasked!")

    print("Test accuracy after fine-tuning: ", get_bleu_score(model, tokenizer, test_loader, MBPP_Test_DF))

    if args.save_results:
        directory = f"./sparse_models/T5ForConditionalGeneration/{args.exp_name}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model.state_dict(), f'./sparse_models/T5ForConditionalGeneration/{args.exp_name}/sparse_weights_finetuned.pth')


def main():
    """
    Main function to handle dynamic pruning using a genetic algorithm.
    This function parses command-line arguments to determine the pruning
    and fine-tuning options for the CodeT5 model. It supports pruning specific
    layers or all layers, and optionally fine-tuning the pruned model. Results
    can also be saved to disk.
    Command-line arguments:
    --layer: str, optional
        Name of the layer to prune (do not specify to prune all layers).
    --finetune: bool, optional
        Finetune the pruned model if specified.
    --save_results: bool, optional
        Store results on disk if specified (do not specify if experimenting).
    --exp_name: str, optional
        Name of the experiment (default is 'exp1').
    The function calls `fine_tune_codeT5()` if the `--finetune` flag is set.
    Otherwise, it calls `prune_codeT5()` with the specified layer or `None` to prune all layers.
    """
    
    global args

    parser = argparse.ArgumentParser(description='dynamic pruning using genetic algorithm')
    parser.add_argument('--layer', default=None, type=str, help='name of the layer to prune (do not specify to prune all layers)')
    parser.add_argument('--finetune', action='store_true', help='finetune pruned model')
    parser.add_argument('--save_results', action='store_true', help='store results on disk (do not specify if experimenting)')
    parser.add_argument('--exp_name', default='exp1', type=str, help='experiment name')
    args = parser.parse_args()

    if args.finetune:
        fine_tune_codeT5()
    else:
        if args.layer:
            prune_codeT5(args.layer)
        else:
            prune_codeT5(None)
    

if __name__ == "__main__" :
    main()
    