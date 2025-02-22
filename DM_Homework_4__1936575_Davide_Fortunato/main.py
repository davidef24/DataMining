import random
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer




# Constants
HIDDEN_DIM = 16
DROPOUT = 0.4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 300

def visualize_embeddings(embeddings, labels, title="Embeddings Visualized with t-SNE"):
    """
    Visualize embeddings using t-SNE.

    Args:
        embeddings (torch.Tensor or np.ndarray): Node embeddings of shape (num_nodes, embedding_dim).
        labels (torch.Tensor or list): Labels corresponding to the nodes for coloring.
        title (str): Title for the plot.
    """
    # Ensure embeddings are in NumPy format
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot the 2D embeddings
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="Set1",
        alpha=0.7,
    )
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()



def generate_node2vec_embeddings(graph, dimensions=256, walk_length=20, num_walks=60, workers=2):
    """
    Generate Node2Vec embeddings for a given graph.

    Args:
        graph (networkx.Graph): Input graph.
        dimensions (int): Dimensions of embeddings.
        walk_length (int): Length of random walks.
        num_walks (int): Number of random walks per node.
        workers (int): Number of workers for parallel processing.

    Returns:
        torch.Tensor: Generated Node2Vec embeddings.
    """
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=8, min_count=1, batch_words=8)
    embeddings = torch.tensor([model.wv[str(node)] for node in graph.nodes()])
    return embeddings

class NodeClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NodeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
# Define the Node Classifier with log_softmax

def compute_edge_embeddings(embeddings, edges, method="concatenate"):
    """
    Compute edge embeddings from node embeddings.

    Args:
        embeddings (torch.Tensor): Node embeddings.
        edges (np.ndarray): Edge list as a NumPy array of shape (num_edges, 2).
        method (str): Method to compute edge embeddings ("concatenate", "elementwise_product", "absolute_difference").

    Returns:
        np.ndarray: Edge embeddings.
    """
    edge_embeddings = []
    for u, v in edges:
        u_emb, v_emb = embeddings[u], embeddings[v]
        if method == "concatenate":
            edge_emb = np.concatenate([u_emb, v_emb])
        elif method == "elementwise_product":
            edge_emb = u_emb * v_emb
        elif method == "absolute_difference":
            edge_emb = np.abs(u_emb - v_emb)
        else:
            raise ValueError(f"Unknown method: {method}")
        edge_embeddings.append(edge_emb)
    return np.array(edge_embeddings)

def link_prediction_n2v(embeddings, edges, method="absolute_difference"):
    """
    Perform link prediction using embeddings and a classifier.

    Args:
        embeddings (torch.Tensor): Node embeddings.
        edges (tuple): Tuple of positive and negative edge lists.
        labels (np.ndarray): Labels for the edges (1 for positive, 0 for negative).
        method (str): Method for computing edge embeddings.

    Returns:
        dict: Evaluation metrics.
    """
    positive_edges, negative_edges = edges
    positive_embeddings = compute_edge_embeddings(embeddings.numpy(), positive_edges, method=method)
    negative_embeddings = compute_edge_embeddings(embeddings.numpy(), negative_edges, method=method)

    X = np.vstack([positive_embeddings, negative_embeddings])
    y = np.hstack([np.ones(len(positive_edges)), np.zeros(len(negative_edges))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    roc_auc = roc_auc_score(y_test, y_pred)

    return {"Accuracy": accuracy, "ROC AUC": roc_auc}


def train_node_classifier(embeddings, labels, epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Train a node classifier on given embeddings and labels.

    Args:
        embeddings (torch.Tensor): Node embeddings from node2vec
        labels (torch.Tensor): Node labels.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.

    Returns:
        NodeClassifier: Trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.3, random_state=42)
    model = NodeClassifier(input_dim=embeddings.shape[1], output_dim=len(torch.unique(labels)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(X_train.float())
        loss = criterion(output, y_train.long())
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return model, X_test, y_test

def evaluate_node_classifier(model, X_test, y_test):
    """
    Evaluate a trained node classifier.

    Args:
        model (NodeClassifier): Trained classifier.
        X_test (torch.Tensor): Test node embeddings.
        y_test (torch.Tensor): Test labels.

    Returns:
        dict: Dictionary containing accuracy and ROC AUC.
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.float())
        y_pred_probs = torch.softmax(y_pred, dim=1)
        _, predicted_classes = torch.max(y_pred_probs, dim=1)

        accuracy = accuracy_score(y_test.numpy(), predicted_classes.numpy())

        roc_auc = roc_auc_score(
            torch.nn.functional.one_hot(y_test).numpy(),
            y_pred_probs.numpy(),
            multi_class="ovr",
        )

        return {"Accuracy": accuracy, "ROC AUC": roc_auc}


def prepare_edges(data, test_ratio=0.3, val_ratio=0.15):
    """
    Prepares positive and negative edges and splits them into train, validation, and test sets.

    Parameters:
        data: PyG dataset object.
        test_ratio: Proportion of edges to allocate for testing.
        val_ratio: Proportion of edges to allocate for validation.

    Returns:
        splits: A dictionary with train, val, and test edges and labels.
    """
    # Positive edges
    positive_edges = data.edge_index

    # Negative edges
    negative_edges = negative_sampling(
        edge_index=positive_edges,
        num_nodes=data.num_nodes,
        num_neg_samples=positive_edges.size(1)
    )

    # Combine edges and create labels
    edge_label_index = torch.cat([positive_edges, negative_edges], dim=1)
    edge_labels = torch.cat([
        torch.ones(positive_edges.size(1)),  # Positive label
        torch.zeros(negative_edges.size(1))  # Negative label
    ], dim=0)

    # Train-validation-test split
    train_idx, test_idx = train_test_split(
        torch.arange(edge_labels.size(0)),
        test_size=test_ratio,
        random_state=42
    )
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    splits = {
        "train": (edge_label_index[:, train_idx], edge_labels[train_idx]),
        "val": (edge_label_index[:, val_idx], edge_labels[val_idx]),
        "test": (edge_label_index[:, test_idx], edge_labels[test_idx]),
    }

    return splits

class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Initializes a GNN for link prediction.

        Parameters:
            in_channels: Number of input features per node.
            hidden_channels: Number of hidden units in GNN layers.
            out_channels: Dimensionality of output embeddings.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        """Encodes node features into embeddings."""
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode_dot(self, z, edge_label_index):
        """Predicts edge scores using the dot product."""
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)


# Helper functions
def load_dataset(name, root):
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    return dataset, data

# Utility Functions
def generate_masks(data, train_ratio=0.6, val_ratio=0.2):
    num_nodes = data.num_nodes
    indices = list(range(num_nodes))
    random.shuffle(indices)

    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask

def assign_masks(data):
    train_mask, val_mask, test_mask = generate_masks(data)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

def print_mask_summary(dataset_name, train_mask, val_mask, test_mask):
    print(f"{dataset_name} Train Mask: {train_mask.sum()} nodes")
    print(f"{dataset_name} Validation Mask: {val_mask.sum()} nodes")
    print(f"{dataset_name} Test Mask: {test_mask.sum()} nodes")

def resize_weights(state_dict, target_dim, layer_name, new_shape):
    """
    Resize the weights or biases for a layer in the state_dict to match the target dimensions.

    Parameters:
        state_dict (dict): Model's state dictionary containing weights and biases.
        target_dim (int): Target dimension (number of classes or features).
        layer_name (str): Name of the layer to resize.
        new_shape (tuple): Shape of the resized tensor.
    """
    # Create a new tensor with the specified shape and initialize it
    new_weight = torch.randn(new_shape) * 0.01  # Small random values for initialization
    original_shape = state_dict[layer_name].shape

    # Determine overlapping dimensions
    overlap_rows = min(new_shape[0], original_shape[0])
    overlap_cols = min(new_shape[1], original_shape[1])

    # Copy overlapping weights
    new_weight[:overlap_rows, :overlap_cols] = state_dict[layer_name][:overlap_rows, :overlap_cols]

    # Update the state_dict
    state_dict[layer_name] = new_weight

#adjust weights and biases of a pre-trained model to fit a new dataset with potentially different input/output features
def resize_bias(state_dict, target_dim, layer_name):
    if state_dict[layer_name].shape[0] != target_dim:
        print(f"Resizing {layer_name}: {state_dict[layer_name].shape} -> ({target_dim},)")
        new_bias = torch.zeros(target_dim)
        overlap_dim = min(state_dict[layer_name].shape[0], target_dim)
        new_bias[:overlap_dim] = state_dict[layer_name][:overlap_dim]
        state_dict[layer_name] = new_bias

def evaluate_cross_dataset(model, data, device):
    model.eval()
    data = data.to(device)
    output = model(data.x, data.edge_index)
    predictions = output.argmax(dim=1).cpu()
    true_labels = data.y.cpu()
    report = classification_report(true_labels[data.test_mask], predictions[data.test_mask], digits=4)
    return report

def load_and_adjust_weights(pretrained_path, dataset, hidden_dim):
    state_dict = torch.load(pretrained_path)
    resize_weights(state_dict, dataset.num_features, 'conv1.lin.weight', (hidden_dim, dataset.num_features))
    resize_bias(state_dict, hidden_dim, 'conv1.bias')
    resize_weights(state_dict, dataset.num_classes, 'conv2.lin.weight', (dataset.num_classes, hidden_dim))
    resize_bias(state_dict, dataset.num_classes, 'conv2.bias')
    return state_dict



# GCN Model Definition
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
def train_link(model, data, splits, n_epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Trains the GNN model for link prediction.

    Parameters:
        model: GNN model instance.
        data: PyG dataset object.
        splits: Dictionary containing train, val, and test splits.
        n_epochs: Number of training epochs.
        lr: Learning rate.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_edge_index, train_edge_labels = splits["train"]
    val_edge_index, val_edge_labels = splits["val"]

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Encode node embeddings
        z = model.encode(data.x, data.edge_index)
        out = model.decode_dot(z, train_edge_index)

        #print(out)
        # Compute loss
        loss = criterion(out, train_edge_labels)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model.decode_dot(z, val_edge_index)
            val_loss = criterion(val_out, val_edge_labels)
            val_auc = roc_auc_score(val_edge_labels.cpu(), val_out.sigmoid().cpu())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

def test_link_with_metrics(model, data, splits):
    """
    Evaluates the GNN model on the test set with multiple metrics.

    Parameters:
        model: Trained GNN model.
        data: PyG dataset object.
        splits: Dictionary containing train, val, and test splits.

    Returns:
        metrics: Dictionary of evaluation metrics.
    """
    test_edge_index, test_edge_labels = splits["test"]

    model.eval()
    with torch.no_grad():
        # Encode node embeddings
        z = model.encode(data.x, data.edge_index)

        # Decode test edges
        test_out = model.decode_dot(z, test_edge_index)
        test_probs = test_out.sigmoid()
        test_preds = (test_probs > 0.5).long()

        # Metrics
        metrics = {
            "AUC": roc_auc_score(test_edge_labels.cpu(), test_probs.cpu()),
            "Accuracy": accuracy_score(test_edge_labels.cpu(), test_preds.cpu()),
            "Precision": precision_score(test_edge_labels.cpu(), test_preds.cpu()),
            "Recall": recall_score(test_edge_labels.cpu(), test_preds.cpu()),
            "F1-Score": f1_score(test_edge_labels.cpu(), test_preds.cpu())
        }

    return metrics

# Training and Testing Functions
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index).argmax(dim=1)
    correct = (out[data.test_mask] == data.y[data.test_mask]).sum()
    accuracy = correct.item() / data.test_mask.sum().item()
    return accuracy

# Plot Results
def plot_results(losses, accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(0, len(accuracies) * 10, 10), accuracies, marker='o')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(losses)), losses, color='orange')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def setup_explainer(model, max_neighbors=5):
    """
    Set up the GNNExplainer with the specified model and configuration.

    Args:
        model (torch.nn.Module): The model to explain.
        max_neighbors (int): Maximum number of edges to include in the explanation.

    Returns:
        Explainer: Configured explainer instance.
    """
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=EPOCHS),
        explanation_type='model',
        node_mask_type=None,
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
        threshold_config=dict(
            threshold_type='topk',
            value=max_neighbors,
        ),
    )


def visualize_edge_weights_gnnexplainer(edge_mask, data, node_idx, preds, threshold=0.0, seed=42):
    labels = data.y
    edge_index = data.edge_index

    # Create the graph
    G = nx.Graph()
    for i, edge in enumerate(edge_index.T):
        if edge_mask[i] > threshold:
            G.add_edge(edge[0].item(), edge[1].item(), weight=edge_mask[i])
            G.add_node(edge[0].item(), pred=preds[edge[0].item()])
            G.add_node(edge[1].item(), pred=preds[edge[1].item()])

    edges = G.edges(data=True)
    original_weights = [edge[2]['weight'].item() for edge in edges]
    min_val = min(original_weights)
    max_val = max(original_weights)
    normalized_weights = [(x - min_val) / (max_val - min_val) for x in original_weights]

    node_colors = [int(G.nodes[n]['pred']) for n in G.nodes]

    # Set up matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size as needed
    pos = nx.spring_layout(G, seed=seed)  # Positions for all nodes

    # Draw the graph
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=700,
        node_color=node_colors,
        cmap=plt.get_cmap('rainbow'),
        font_size=10,
        font_weight='bold',
        edge_color=normalized_weights,
        edge_cmap=plt.cm.Blues,
        alpha=0.8,
        ax=ax  # Pass the axis to networkx
    )

    # Add edge labels
    edge_labels = {
        (u, v): f'{w:.3f}' for ((u, v, d), w) in zip(edges, normalized_weights)
    }
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color='black',
        ax=ax  # Pass the axis to networkx
    )

    # Add the title
    ax.set_title(f'GNNExplainer on node {node_idx}: label = {labels[node_idx]}, pred = {preds[node_idx]}', pad=20)

    # Show the plot
    plt.show()


def visualize_edge_weights(edge_mask, data, node_idx, preds, threshold=0.0, seed=42):
    """
    Visualize the edge weights for a given explanation.

    Args:
        edge_mask (torch.Tensor): Edge mask from the explainer.
        data (torch_geometric.data.Data): The input graph data.
        node_idx (int): Index of the node being explained.
        preds (torch.Tensor): Predicted labels for all nodes.
        threshold (float): Minimum edge weight for visualization.
        seed (int): Seed for the graph layout.
    """
    labels = data.y
    edge_index = data.edge_index

    G = nx.Graph()
    for i, edge in enumerate(edge_index.T):
        if edge_mask[i] > threshold:
            G.add_edge(edge[0].item(), edge[1].item(), weight=edge_mask[i])
            G.add_node(edge[0].item(), pred=preds[edge[0].item()])
            G.add_node(edge[1].item(), pred=preds[edge[1].item()])

    edges = G.edges(data=True)
    original_weights = [edge[2]['weight'].item() for edge in edges]
    min_val = min(original_weights)
    max_val = max(original_weights)
    normalized_weights = [(x - min_val) / (max_val - min_val) for x in original_weights]

    node_colors = [int(G.nodes[n]['pred']) for n in G.nodes]

    pos = nx.spring_layout(G, seed=seed)  # Positions for all nodes
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=700,
        node_color=node_colors,
        cmap=plt.get_cmap('rainbow'),
        font_size=10,
        font_weight='bold',
        edge_color=normalized_weights,
        edge_cmap=plt.cm.Blues,
        alpha=0.8
    )

    edge_labels = {(u, v): f'{w:.3f}' for ((u, v, d), w) in zip(edges, normalized_weights)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

    plt.title(f'GNNExplainer on node {node_idx}: label = {labels[node_idx]}, pred = {preds[node_idx]}')
    plt.show()

def explain_and_visualize_nodes(explainer, data, model, target_nodes, threshold=0.0, device='cpu'):
    """
    Explain and visualize predictions for specific nodes in the graph.

    Args:
        explainer (Explainer): Configured explainer instance.
        data (torch_geometric.data.Data): The input graph data.
        model (torch.nn.Module): The model being explained.
        target_nodes (list): List of node indices to explain.
        threshold (float): Minimum edge weight for visualization.
        device (str): Device to run computations on.
    """
    model.eval()
    _, preds = model(data.x, data.edge_index).max(dim=1)

    for node_idx in target_nodes:
        explanation = explainer(data.x.to(device), data.edge_index.to(device), index=node_idx)
        edge_mask = explanation.edge_mask
        visualize_edge_weights_gnnexplainer(edge_mask, data, node_idx, preds, threshold=threshold)

def visualize_embeddings_cora(data, model, title="GNN Node Embeddings with t-SNE"):
  model.eval()
  with torch.no_grad():
    embeddings = model.conv1(data.x, data.edge_index)
    embeddings = model.conv2(embeddings, data.edge_index)
    embeddings = embeddings.cpu().numpy()

  # Apply t-SNE
  tsne = TSNE(n_components=2, random_state=42)
  embeddings_2d = tsne.fit_transform(embeddings)
  
  # Plot
  plt.figure(figsize=(10, 8))
  scatter = plt.scatter(
    embeddings_2d[:, 0], 
    embeddings_2d[:, 1], 
    c=data.y.cpu(), 
    cmap='Set1', 
    alpha=0.7
  )
  plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.title(title)
  plt.xlabel("t-SNE Dimension 1")
  plt.ylabel("t-SNE Dimension 2")
  plt.show()

def validate(model, data):
    model.eval()
    _, pred = model(data.x, data.edge_index).max(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    return acc

def train_cora(model, data, optimizer, criterion, epochs, eval_interval=10):
    """
    Trains and evaluates the model.

    Args:
        model: The GCN model.
        data: The dataset for training and testing.
        optimizer: The optimizer for training.
        criterion: Loss function.
        epochs: Number of training epochs.
        eval_interval: Interval at which to evaluate the model.

    Returns:
        losses, accuracies
    """
    losses, accuracies = [], []
    for epoch in range(epochs):
        loss = train(model, data, optimizer, criterion)
        losses.append(loss)
        if epoch % eval_interval == 0:
            acc = validate(model, data)
            accuracies.append(acc)
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}')
    return losses, accuracies

def evaluate_cora(model, data):
    """
    Evaluates the model's accuracy on the test set.

    Args:
        model: The GCN model.
        data: The dataset for evaluation.

    Returns:
        Test accuracy.
    """
    test_acc = test(model, data)
    print(f'Test Accuracy: {test_acc:.4f}')
    return test_acc



def generate_classification_report(model, data):
    """
    Generates a classification report for the model.

    Args:
        model: The trained GCN model.
        data: The dataset for evaluation.

    Returns:
        None
    """
    model.eval()
    out = model(data.x, data.edge_index).argmax(dim=1).cpu()
    true = data.y.cpu()
    report = classification_report(true[data.test_mask], out[data.test_mask], digits=4)
    print(report)


# Main Execution
def main():
    # Load Dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    cora_path = 'data/Cora'
    citeseer_path = 'data/CiteSeer'
    pubmed_path = 'data/PubMed'

    dataset, cora_data = load_dataset('Cora', cora_path)
    citeseer, citeseer_data = load_dataset('CiteSeer', citeseer_path)
    pubmed, pubmed_data = load_dataset('PubMed', pubmed_path)


    # Assign masks
    assign_masks(cora_data)
    assign_masks(citeseer_data)
    assign_masks(pubmed_data)

    print_mask_summary("CiteSeer", citeseer_data.train_mask, citeseer_data.val_mask, citeseer_data.test_mask)
    print_mask_summary("PubMed", pubmed_data.train_mask, pubmed_data.val_mask, pubmed_data.test_mask)

    # Prepare Model and Optimizer
    cora_model = GCN(input_dim=dataset.num_features, hidden_dim=HIDDEN_DIM, 
                output_dim=dataset.num_classes, dropout=DROPOUT).to(device)
    cora_data = cora_data.to(device)
    optimizer = optim.Adam(cora_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Train Model
    losses, accuracies = train_cora(model=cora_model, data=cora_data, optimizer=optimizer, 
                                                 criterion=criterion, epochs=EPOCHS, eval_interval=10)

    # Plot Results
    plot_results(losses, accuracies)

    # Evaluate Model
    evaluate_cora(cora_model, cora_data)

    # Classification Report
    generate_classification_report(cora_model, cora_data)

    visualize_embeddings_cora(cora_data, cora_model)

    # Save Model
    torch.save(cora_model.state_dict(), 'gcn_cora.pt')

    # Load and adjust weights for CiteSeer
    pretrained_path = 'gcn_cora.pt'
    hidden_dim = HIDDEN_DIM
    state_dict_citeseer = load_and_adjust_weights(pretrained_path, citeseer, hidden_dim)

    # Model for CiteSeer
    model_citeseer = GCN(
        input_dim=citeseer.num_features,
        hidden_dim=hidden_dim,
        output_dim=citeseer.num_classes,
        dropout=DROPOUT  # Example dropout value
    ).to(device)

    model_citeseer.load_state_dict(state_dict_citeseer)

    # Evaluate CiteSeer
    print("Report for CiteSeer Testing:")
    print(evaluate_cross_dataset(model_citeseer, citeseer_data, device))

    # Load and adjust weights for PubMed
    state_dict_pubmed = load_and_adjust_weights(pretrained_path, pubmed, hidden_dim)

    # Model for PubMed
    model_pubmed = GCN(input_dim=pubmed.num_features, hidden_dim=hidden_dim, output_dim=pubmed.num_classes, dropout=DROPOUT).to(device)
    model_pubmed.load_state_dict(state_dict_pubmed)

    # Evaluate PubMed
    print("Report for PubMed Testing:")
    print(evaluate_cross_dataset(model_pubmed, pubmed_data, device))

    # Prepare edge splits
    splits = prepare_edges(cora_data)

    # Initialize model
    model_link = GNNLinkPredictor(
        in_channels=cora_data.num_features,
        hidden_channels=128,
        out_channels=64
    ).to(device)

    # Train the model
    train_link(model_link, cora_data, splits, n_epochs=EPOCHS, lr=LEARNING_RATE)

    # Test the model and evaluate metrics
    test_metrics = test_link_with_metrics(model_link, cora_data, splits)

    # Print metrics
    print("Test Metrics for Link prediction on Cora dataset:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Prepare data
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(cora_data.edge_index.cpu().numpy().T)

    # Generate embeddings
    n2v_embeddings = generate_node2vec_embeddings(nx_graph)

    visualize_embeddings(n2v_embeddings, cora_data.y)

    # Train node classifier
    model_nc, x_test, y_test = train_node_classifier(n2v_embeddings, cora_data.y, 1000)

    metrics = evaluate_node_classifier(model_nc, x_test, y_test)
    print("Metric for node classification on node2vec embeddings")
    print(metrics)

    # Link prediction
    positive_edges = cora_data.edge_index.T.numpy()
    negative_edges = np.array([
        (u, v) for u, v in np.random.randint(0, nx_graph.number_of_nodes(), (len(positive_edges), 2))
        if not nx_graph.has_edge(u, v)
    ])
    metrics = link_prediction_n2v(n2v_embeddings, (positive_edges, negative_edges))
    print("Metric for link prediction on node2vec embeddings")
    print(metrics)

    # Explainability part
    max_neighbors = 15
    explainer = setup_explainer(cora_model, max_neighbors=max_neighbors)
    target_nodes = [1986, 45, 35, 251]
    explain_and_visualize_nodes(explainer, cora_data, cora_model, target_nodes, threshold=0.0, device=device)

if __name__ == "__main__":
    main()