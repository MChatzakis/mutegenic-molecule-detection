# File: utils.py
# Contains several useful functions for the GNN and data
# Author: Manos Chatzakis (emmanouil.chatzakis@epfl.ch)

import matplotlib.pyplot as plt

import random
import copy

import numpy as np

import torch
import torch.nn as nn

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, roc_curve


def create_adjacency_matrix(edge_index, num_nodes):
    """Generates an adjacency matrix from an edge list.

    Args:
        edge_index (tensor): Edge list tensor
        num_nodes (int): Number of nodes in the graph

    Returns:
        tensor: Tensor of shape (num_nodes, num_nodes) representing the adjacency matrix
    """
    adj = torch.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1
    return adj


def load_MUTAG():
    """
    Loads the MUTAG dataset from HuggingFace datasets. It also creates the adjacency matrix for each graph.

    Returns:
        list: List of MUTAG graphs
    """
    mutag_raw = load_dataset("graphs-datasets/MUTAG")["train"]

    mutag = []
    for graph in mutag_raw:
        adj_matrix = create_adjacency_matrix(graph["edge_index"], graph["num_nodes"])

        # Raw matrix does not have ones in the diagonal
        adj_matrix_raw = copy.deepcopy(adj_matrix)

        ind = np.diag_indices(adj_matrix.shape[0])
        adj_matrix[ind[0], ind[1]] = torch.ones(adj_matrix.shape[0])

        entry = {
            "edge_index": torch.tensor(graph["edge_index"]),
            "node_feat": torch.tensor(graph["node_feat"]),
            "edge_attr": torch.tensor(graph["edge_attr"]),
            "y": torch.tensor(graph["y"]),
            "num_nodes": graph["num_nodes"],
            "adjacency_matrix": adj_matrix,
            "adjacency_matrix_raw": adj_matrix_raw,
        }

        mutag.append(entry)

    return mutag


def remove_mutagenic(mutag, num):
    """Remove mutagenic graphs from the dataset, to ensure balancing of positive and negative samples.

    Args:
        mutag (list): List of MUTAG graphs
        num (int): How many mutagenic graphs to remove

    Returns:
        list: Balanced list of MUTAG graphs
    """
    
    random.shuffle(mutag)
    balanced_mutag = []
    removed = 0

    for graph in mutag:
        if graph["y"][0] == 1 and removed < num:
            removed += 1
        else:
            balanced_mutag.append(graph)

    return balanced_mutag


def load_and_balance_MUTAG():
    """Wrapper to load and balance the MUTAG dataset.

    Returns:
        list: List of MUTAG graphs
    """
    mutag = load_MUTAG()
    
    mutagenic_graphs, nonmutagenic_graphs = count_classes(mutag)
    mutag_data = remove_mutagenic(mutag, mutagenic_graphs - nonmutagenic_graphs)
    
    return mutag_data


def train_val_test(graph_data, train_ratio=0.8, val_ratio=0.1):
    """Splits MUTAG into train, validation, and test sets.

    Args:
        graph_data (list): List of MUTAG graphs
        train_ratio (float, optional): Ratio of train sets. Defaults to 0.8.
        val_ratio (float, optional): Ratio of validation sets. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    num_graphs = len(graph_data)

    num_train = int(num_graphs * train_ratio)
    num_val = int(num_graphs * val_ratio)
    num_test = num_graphs - num_train - num_val

    random.shuffle(graph_data)

    return (
        graph_data[:num_train],
        graph_data[num_train : num_train + num_val],
        graph_data[num_train + num_val :],
    )


def count_classes(mutag):
    """Count mutagenic and non-mutagenic graphs in the dataset.

    Args:
        mutag (list): MUTAG graphs

    Returns:
        tuple(int,int): Number of mutagenic and non-mutagenic graphs
    """
    pos = 0
    neg = 0
    for graph in mutag:
        if graph["y"][0] == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg


def fair_train_test_split(mutag, train_size=0.7, test_size=0.3, random_state=42):
    """Split the dataset into train, validation, and test sets, ensuring that the ratio of positive and negative samples is the same in all sets.

    Args:
        mutag (list): List of MUTAG graphs
        train_size (float, optional): Ratio for train samples. Defaults to 0.7.
        test_size (float, optional): Ratio for test samples. Defaults to 0.3.
        random_state (int, optional): seed. Defaults to 42.

    Returns:
        tuple(Train, Val, Test): Train, validation, and test sets
    """
    
    # Extract the labels from the objects
    y = []
    for obj in mutag:
        y.append(int(obj["y"][0]))

    # Split the data into a training set (70%), validation set (15%), and test set (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        mutag, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test


def plot_training_loop_summary(results, path=None):
    """Plot the training loop summary.

    Args:
        results (dict): Per epoch results
        path (str, optional): Directory to save the graph. Defaults to None.
    """
    # Plot the loss
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].plot(results["train_loss"], label="Train Loss", color="red")
    axs[0].plot(results["val_loss"], label="Validation Loss", color="blue")
    axs[0].set_title("Losses")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(results["train_accuracy"], label="Train Accuracy", color="red")
    axs[1].plot(results["val_accuracy"], label="Validation Accuracy", color="blue")
    axs[1].set_title("Accuracies")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    axs[2].plot(results["train_f1_score"], label="Train F1 Score", color="red")
    axs[2].plot(results["val_f1_score"], label="Validation F1 Score", color="blue")
    axs[2].set_title("F1 Scores")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("F1 Score")
    axs[2].legend()

    fig.tight_layout()

    if path is not None:
        plt.savefig(path)
        print("Saved plot to", path)
    else:
        plt.show()

    plt.close()


def plot_model_comparison_results(accuracies, f1_scores, labels, path=None):
    """Plot the results of the model comparisons.

    Args:
        accuracies (list): List of per-model accuracies
        f1_scores (list): List of per-model F1 scores
        labels (list): List of model names
        path (str, optional): Path to save the graph. Defaults to None.
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))

    axs[0].set_xticks(np.arange(len(labels)))
    axs[0].set_xticklabels(labels, rotation=45)
    axs[1].set_xticks(np.arange(len(labels)))
    axs[1].set_xticklabels(labels, rotation=45)

    for i, model_name in enumerate(labels):
        axs[0].bar(model_name, accuracies[i], color="gray", alpha=0.5)
        axs[1].bar(model_name, f1_scores[i], color="red", alpha=0.5)

    #axs[0].set_title("Accuracy")
    #axs[0].set_xlabel("Model")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_ylim([0.4, 0.85])

    #axs[1].set_title("F1 Score")
    #axs[1].set_xlabel("Model")
    axs[1].set_ylabel("F1 Score")
    axs[1].set_ylim([0.4, 0.85])

    # Rotate the x-labels in both subplots

    if path is not None:
        plt.savefig(path)

    fig.tight_layout()


def plot_model_comparison_subfigs(plotting_data):
    fig, axs = plt.subplots(len(plotting_data), 2, figsize=(8, 10))
    
    for index, exp_type in enumerate(plotting_data.keys()):
        accuracies = plotting_data[exp_type][0]
        f1_scores = plotting_data[exp_type][1]
        labels = plotting_data[exp_type][2]
        
        axs[index][0].set_xticks(np.arange(len(labels)))
        axs[index][0].set_xticklabels(labels, rotation=45)
        axs[index][1].set_xticks(np.arange(len(labels)))
        axs[index][1].set_xticklabels(labels, rotation=45)
        
        for i, model_name in enumerate(labels):
            axs[index][0].bar(model_name, accuracies[i], color="gray", alpha=0.5)
            axs[index][1].bar(model_name, f1_scores[i], color="red", alpha=0.5)
        
        axs[index][0].set_ylabel(f"{exp_type}: Accuracy")
        axs[index][1].set_ylabel(f"{exp_type}: F1 Score")
        
        axs[index][0].set_ylim([0.4, 0.8])
        axs[index][1].set_ylim([0.4, 0.8])
    
    axs[0][0].set_title("Accuracy Comparison")
    axs[0][1].set_title("F1 Score Comparison")
    
    axs[len(plotting_data)-1][0].set_xlabel("Model")
    axs[len(plotting_data)-1][1].set_xlabel("Model")
    
    fig.tight_layout()
        
        

def avg_pool(X):
    """Take the mean of a tensor of size (nodes, features), transofrming it to (1, features).

    Args:
        X (tensor): Tensor of size (nodes, features)

    Returns:
        tensor: Averaged tensor of size (1, features)
    """
    dim = -1 if X.dim() == 1 else -2
    return X.mean(dim=dim, keepdim=X.dim() <= 2)


class BaselineRandomModel(nn.Module):
    """Baseline model that does random predictions."""
    
    def __init__(self):
        super().__init__()

    def forward(self, X, adj):
        return X

    def predict_class(self, X, adj):
        class_index = random.choice([0, 1])
        return class_index

    def predict_class_from_logits(self, logits):
        class_index = random.choice([0, 1])
        return class_index


def evaluate_random_baseline(dataloader):
    """Evaluate the random baseline model.

    Args:
        dataloader (dataloader): _description_

    Returns:
        _type_: _description_
    """
    random.seed(42)
    predicted_classes = []
    true_classes = []
    for data in dataloader:
        gold_label = (data["y"])[0]
        pred_class = random.choice([0, 1])

        predicted_classes.append(int(pred_class))
        true_classes.append(int(gold_label))

    test_acc = np.mean(np.array(predicted_classes) == np.array(true_classes))
    test_f1_score = f1_score(np.array(true_classes), np.array(predicted_classes))

    return test_acc, test_f1_score


def generate_per_node_edge_features(mutag_graph):
    """Generate the per-node edge features for a MUTAG graph.

    Args:
        mutag_graph (dict): MUTAG graph data

    Returns:
        tensor: Edge features of size (nodes, edge_features)
    """
    
    edge_list = mutag_graph["edge_index"]
    edge_attr = mutag_graph["edge_attr"]
    adj = mutag_graph["adjacency_matrix_raw"]

    per_node_feats = []

    for i in range(adj.shape[0]):
        node_edge_feats = []
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                for z in range(edge_list.shape[1]):
                    if edge_list[0][z] == i and edge_list[1][z] == j:
                        node_edge_feats.append(edge_attr[z])

        if len(node_edge_feats) == 0:
            node_edge_feats.append(torch.zeros(4))

        mean = torch.mean(torch.stack(node_edge_feats, dim=0), dim=0)
        per_node_feats.append(mean)

    return torch.stack(per_node_feats, dim=0)
