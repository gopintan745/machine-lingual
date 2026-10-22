# This file contains functions for visualizing the evaluation results of the machine translation models. 
# It includes functions to plot chrF++ scores, loss curves, and other relevant metrics to assess the performance of the models on different datasets and language pairs.
import matplotlib.pyplot as plt
import numpy as np

def plot_chrf_scores(chrf_scores, title="chrF++ Scores", xlabel="Language Pairs", ylabel="chrF++ Score"):
    """
    Plots the chrF++ scores for different language pairs.

    Args:
        chrf_scores (dict): A dictionary where keys are language pairs and values are chrF++ scores.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    language_pairs = list(chrf_scores.keys())
    scores = list(chrf_scores.values())

    plt.figure(figsize=(10, 6))
    plt.bar(language_pairs, scores, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)  # chrF++ scores range from 0 to 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"chrf_scores_plot_{language_pairs}.png")  # Save the plot as a PNG file
    plt.show()


def plot_loss_curve(train_losses, val_losses, title="Training and Validation Loss", xlabel="Epochs", ylabel="Loss", language_pairs:str|None = None):
    """
    Plots the training and validation loss curves.

    Args:
        train_losses (list): A list of training loss values.
        val_losses (list): A list of validation loss values.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        language_pairs(str |None): The language pair of the dataset being trained and evaluated.
    """
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(epochs)
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"loss_curve_plot_{language_pairs}")
    plt.show()



def plot_optimization_results(trial_values, title="Hyperparameter Optimization Results", xlabel="Trial Number", ylabel="Evaluation Loss", language_pairs:str|None = None):
    """
    Plots the results of hyperparameter optimization trials.

    Args:
        trial_values (list): A list of evaluation loss values from different trials.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        language_pairs(str |None): The language pair of the dataset being trained and evaluated.
    """
    trials = np.arange(1, len(trial_values) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(trials, trial_values, marker='o', color='purple')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(trials)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"optimization_results_{language_pairs}")
    plt.show()


def plot_hyperparameter_importance(importance_dict, title="Hyperparameter Importance", xlabel="Hyperparameters", ylabel="Importance Score", language_pair:str|None =None):
    """
    Plots the importance of different hyperparameters based on their impact on the evaluation loss.

    Args:
        importance_dict (dict): A dictionary where keys are hyperparameter names and values are their importance scores.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        language_pairs(str |None): The language pair of the dataset being trained and evaluated.
    """
    hyperparameters = list(importance_dict.keys())
    importance_scores = list(importance_dict.values())

    plt.figure(figsize=(10, 6))
    plt.bar(hyperparameters, importance_scores, color='orange')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"hyperparameter_importance_{language_pair}")
    plt.show()


def plot_chrf_vs_llm_jugdement(chrf_scores, llm_judgement_scores, title="chrF++ vs LLM Judgement Scores", xlabel="Language Pairs", ylabel="Scores", language_pairs:str|None = None):
    """
    Plots the chrF++ scores against LLM judgement scores for different language pairs.

    Args:
        chrf_scores (dict): A dictionary where keys are language pairs and values are chrF++ scores.
        llm_judgement_scores (dict): A dictionary where keys are language pairs and values are LLM judgement scores.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        language_pairs(str |None): The language pair of the dataset being trained and evaluated.
    """
    language_pairs = list(chrf_scores.keys())
    chrf_values = list(chrf_scores.values())
    llm_values = [llm_judgement_scores.get(pair, 0) for pair in language_pairs]

    plt.figure(figsize=(10, 6))
    plt.scatter(chrf_values, llm_values, color='green')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()