# File: train_utils.py
# Contains the training utilities for the GNN model
# Author: Manos Chatzakis (emmanouil.chatzakis@epfl.ch)

import numpy as np
import pandas as pd

import torch

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm


def train_step(
    data,
    model,
    loss_fn,
    optimizer,
    train_true_labels,
    train_predicted_labels,
    features_key="node_feat",
):
    """Train step for a neural network model.

    Args:
        data (dict): A MUTAG graph
        model (gnn): A GNN model
        loss_fn : Loss function to use
        optimizer : Optimizer to use
        train_true_labels (list): List to append the true labels
        train_predicted_labels (list): List to append the predicted labels
        features_key (str, optional): Which features to use. Defaults to "node_feat".

    Returns:
        loss: Float representing the loss
    """

    # Set model to train mode
    model.train()

    # Compute prediction and loss
    x = data[features_key][0]
    y = data["y"][0]
    adj = data["adjacency_matrix"][0]

    logits = model(x, adj)
    loss = loss_fn(logits, y)

    # 1. Set the gradient of trainable parameters to 0.
    optimizer.zero_grad()

    # 2. Automatically calculate the gradient of trainable parameters.
    loss.backward()

    # 3. Automatically update the trainable parameters using the gradient.
    optimizer.step()

    # 4. Clean the gradient of trainable parameters.
    optimizer.zero_grad()

    # 5. Update the training metrics
    pred_class = predict_class_from_logits(logits)
    train_predicted_labels.append(int(pred_class))
    train_true_labels.append(int(y))

    return loss.item()


def test(dataloader, model, criterion, features_key="node_feat"):
    """Test function computing the performance of a GNN model.

    Args:
        dataloader (dataloader): PyTorch dataloader to be used
        model (GNN model): GNN model to evaluate
        criterion: Loss function to use
        features_key (str, optional): Which features to use. Defaults to "node_feat".

    Returns:
        tuple: A tuple containing the accuracy, loss, F1-score and predicted of classes
    """

    # Set the model to evaluation mode
    model.eval()

    # Iterate over the dataloader
    predicted_classes = []
    true_classes = []
    losses = []
    for data in dataloader:
        feats = data[features_key][0]
        adj = data["adjacency_matrix"][0]

        logits = model(feats, adj)

        gold_label = (data["y"])[0]
        loss = criterion(logits, gold_label)
        losses.append(loss.item())

        pred_class = predict_class_from_logits(logits)

        # Update the predicted and true classes
        predicted_classes.append(int(pred_class))
        true_classes.append(int(data["y"]))

    # Compute metrics
    test_acc = np.mean(np.array(predicted_classes) == np.array(true_classes))
    test_loss = np.mean(losses)
    test_f1_score = f1_score(np.array(true_classes), np.array(predicted_classes))

    pos = predicted_classes.count(1)
    neg = predicted_classes.count(0)

    return test_acc, test_loss, test_f1_score, pos, neg


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    epochs,
    criterion,
    optimizer,
    verbose=True,
    features_key="node_feat",
    scheduler=None,
):
    """Training loop of GNN model.

    Args:
        model (GNN model): The GNN model to train
        train_dataloader (PyTorch Dataloader): Dataloader of the training data
        val_dataloader (PyTorch Dataloader): Dataloader of the validation data
        epochs (int): How many epochs to train
        criterion : Loss function to use
        optimizer : Optimizer to use
        verbose (bool, optional): Flag to enable logging. Defaults to True.
        features_key (str, optional): Which features to use. Defaults to "node_feat".
        scheduler (optional): Pytorch scheduler to manage the learning rate of the optimizer. Defaults to None.

    Returns:
        results: dict of the per-epoch training results
    """
    
    # Initialize the results dictionary
    results = {
        "train_accuracy": [],
        "val_accuracy": [],
        "train_loss": [],
        "val_loss": [],
        "train_f1_score": [],
        "val_f1_score": [],
    }

    # Start the training loop
    for e in tqdm(range(epochs), desc="Training GNN", unit="epoch"):
        epoch_training_loss_list = []
        train_predicted_labels = []
        train_true_labels = []

        # Iterate over the dataloader
        for batch in train_dataloader:
            train_loss = train_step(
                batch,
                model,
                criterion,
                optimizer,
                train_true_labels,
                train_predicted_labels,
                features_key,
            )
            epoch_training_loss_list.append(train_loss)

        # Apply scheduler if available
        if scheduler is not None:
            scheduler.step()

        # Calculate the per epoch results
        train_loss = np.mean(epoch_training_loss_list)

        val_acc, val_loss, val_f1_score, _, _ = test(
            val_dataloader, model, criterion, features_key
        )

        train_accuracy = np.mean(
            np.array(train_predicted_labels) == np.array(train_true_labels)
        )
        train_f1_score = f1_score(train_true_labels, train_predicted_labels)

        # Update the results dictionary
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["train_accuracy"].append(train_accuracy)
        results["val_accuracy"].append(val_acc)
        results["train_f1_score"].append(train_f1_score)
        results["val_f1_score"].append(val_f1_score)

        # Logging
        if verbose and (e % 50 == 0 or e == epochs - 1):
            print(
                f"{e}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy {train_accuracy:.4f}, Val Accuracy: {val_acc:.4f}, Train F1 Score: {train_f1_score:.4f}, Val F1 Score: {val_f1_score:.4f}"
            )

    return results


def predict_class(model, X, adj):
    """Perform a forward pass and return the predicted class.

    Args:
        model (GNN model): The GNN model that will be used
        X (tensor): Input features tensor of size (nodes, features)
        adj (tensor): Adjacency matrix tensor of size (nodes, nodes)

    Returns:
        torch(int): Class index
    """
    logits = model.forward(X, adj)
    probs = torch.softmax(logits, dim=1)
    
    return torch.argmax(probs)


def predict_class_from_logits(logits):
    """Predict the class from the logits.

    Args:
        logits (tensor): Logit tensor of size (1,classes)

    Returns:
        torch(int): Class index
    """
    probs = torch.softmax(logits, dim=1)
    return torch.argmax(probs)
