import os
import copy
import time
from pathlib import Path

import torch
import numpy as np
import torch_geometric
from torch_geometric.nn import GINConv
from datasail.sail import datasail
from glycowork.glycan_data.loader import HashableDict, lib

from glycontact.process import get_all_clusters_frequency, get_structure_graph


def get_all_structure_graphs(glycan, stereo=None, libr=None):
    """
    Get all structure graphs for a given glycan.
    
    Args:
        glycan (str): The glycan name.
        stereo (str, optional): The stereochemistry. If None, both alpha and beta are returned.
        libr (HashableDict, optional): A library of structures. If None, the default library is used.
    
    Returns:
        list: A list of tuples containing the PDB file name and the corresponding structure graph.
    """
    libr = HashableDict(libr)
    if stereo is None:
        return get_all_structure_graphs(glycan, "alpha", libr) + get_all_structure_graphs(glycan, "beta", libr)
    glycan_path = Path("glycans_pdb") / glycan
    matching_pdbs = [glycan_path / pdb for pdb in sorted(os.listdir(glycan_path)) if stereo in pdb]
    return [(pdb, get_structure_graph(glycan, libr=libr, example_path=pdb)) for pdb in matching_pdbs]


def node2y(attr):
    """
    Extract ML task labels from node attributes.

    Args:
        attr (dict): Node attributes.
    
    Returns:
        list: A list of labels for the node. If all labels are zero, returns None.
    """
    output = [
        attr.get("phi_angle", 0), 
        attr.get("psi_angle", 0), 
        attr.get("SASA", 0), 
        attr.get("flexibility", 0), 
    ]
    if output == [0, 0, 0, 0]:
        return None
    return output


def graph2pyg(g, weight, iupac, conformer):
    """
    Convert a structure graph to a PyTorch Geometric Data object.
    
    Args:
        g (networkx.Graph): The structure graph.
        weight (float): The weight of the graph.
        iupac (str): The IUPAC name of the glycan.
        conformer (str): The conformer name.
    
    Returns:
        torch_geometric.data.Data: The PyTorch Geometric Data object.
    """
    x, y = [], []
    for n in range(len(g.nodes)):
        x.append(lib.get(g.nodes[n]["string_labels"], 0))
        y.append(labels := node2y(g.nodes[n]))
        if labels is None:  # Skip if all labels are zero, i.e., the graph is invalid or broken
            return None
    edge_index = [], []
    for edge in g.edges():
        edge_index[0].append(edge[0])
        edge_index[1].append(edge[1])
        # for bidirectionality
        edge_index[0].append(edge[1])
        edge_index[1].append(edge[0])
    return torch_geometric.data.Data(
        x=torch.tensor(x),
        y=torch.tensor(y),
        edge_index=torch.tensor(edge_index).long(),
        weight=weight,
        iupac=iupac,
        conformer=conformer,
    )


def create_dataset(fresh: bool = True):
    """
    Create a dataset of PyTorch Geometric Data objects from the structure graphs of glycans.

    Args:
        fresh (bool): If True, fetches the latest data. If False, uses cached data.
    
    Returns:
        tuple: A tuple containing the training and testing datasets.
    """
    # Get all clusters and their frequencies.
    data = {}
    for iupac, freqs in get_all_clusters_frequency(fresh=fresh).items():
        try:
            pygs = []
            broken = False
            graphs = get_all_structure_graphs(iupac, None, lib)
            for pathname, graph in graphs:
                # Get the weight of the graph based on the cluster frequency and convert the graph to a PyG Data object.
                weight = freqs[int(pathname.stem.split("_")[0].replace("cluster", ""))]
                pyg = graph2pyg(graph, weight, iupac, pathname.stem)
                if pyg is None:
                    print(f"{iupac}, Conformer {pathname.stem} is None")
                    broken = True
                    break
                pygs.append((pyg, graph))
            if broken:
                # if one conformer is broken, skip the whole glycan
                continue
            # Normalize the weights of the graphs and assign them to the PyG Data objects.
            weights = np.array([pyg.weight for pyg, _ in pygs])
            weights = weights / np.sum(weights)
            for (pyg, _), weight in zip(pygs, weights):
                pyg.weight = torch.tensor([weight])
            data[iupac] = pygs
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error for {iupac}: {e}")

    # Split the data into training and testing sets using DataSAIL. The glycans are weights based on their number of conformers.
    e_splits, _, _ = datasail(
        techniques=["I1e"],
        names=["train", "test"],
        splits=[8, 2],
        e_type="O",
        e_data=[(d, d) for d in data.keys()],
        e_weights={d: len(c) for d, c in data.items()},
    )

    # Create the training and testing datasets.
    train, test = [], []
    for name, split in e_splits["I1e"][0].items():
        if split == "train":
            train.extend(data[name])
        elif split == "test":
            test.extend(data[name])
    return train, test


class VonMisesSweetNet(torch.nn.Module):
    def __init__(
            self, 
            lib_size: int, # number of unique tokens for graph nodes
            num_classes: int = 1, # number of output classes (>1 for multilabel)
            hidden_dim: int = 128, # dimension of hidden layers
            num_components: int = 5 # number of components in the mixture models
        ) -> None:
        "given glycan graphs as input, predicts properties via a graph neural network"
        super(VonMisesSweetNet, self).__init__()
        # Node embedding
        self.item_embedding = torch.nn.Embedding(num_embeddings=lib_size+1, embedding_dim=hidden_dim)

        # Output layers for mixture model parameters
        self.num_components = num_components
        self.num_classes = num_classes  # Currently ignored

        # Convolution operations on the graph (Backbone)
        self.body = torch.nn.Sequential(
            GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
            )),
            GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
            )),
            GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
            )),
        )

        # Classification head for von Mises-distributed properties (phi and psi)
        self.head_von_mises = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.BatchNorm1d(hidden_dim // 2),
            torch.nn.Dropout(p=0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 2),
            torch.nn.BatchNorm1d(hidden_dim // 2),
        )

        # For each torsion angle (phi, psi), predict mixture weights, means, and kappas
        self.fc_weights_von_mises = torch.nn.Linear(hidden_dim // 2, 2 * num_components)  # Logits for mixture weights
        self.fc_means_von_mises = torch.nn.Linear(hidden_dim // 2, 2 * num_components)  # Mean angles
        self.fc_kappas_von_mises = torch.nn.Linear(hidden_dim // 2, 2 * num_components)  # Concentration parameters

        # Classification head for Gaussian-distributed properties (SASA and flexibility)
        self.head_values = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 4),
            torch.nn.BatchNorm1d(hidden_dim // 4),
            torch.nn.Dropout(p=0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 4, 2),
            torch.nn.BatchNorm1d(2),
        )
    
    def predict_von_mises_parameters(self, x, head, fc_weights, fc_means, fc_kappas):
        """
        Predict mixture parameters for a given input tensor.

        Args:
            x: Input tensor [batch_size, hidden_dim]
            head: Head module for the mixture model
            fc_weights: Fully connected layer for weights
            fc_means: Fully connected layer for means
            fc_kappas: Fully connected layer for kappas
        
        Returns:
            Tuple of 
            weights_logits: Logits for mixture weights [batch_size, 2, num_components]
            means: Mean angles in degrees [batch_size, 2, num_components]
            kappas: Concentration parameters [batch_size, 2, num_components]
        """
        x = head(x)
        weights_logits = fc_weights(x)  # [batch_size, 2 * num_components]
        means = fc_means(x)  # [batch_size, 2 * num_components]
        kappas_raw = fc_kappas(x)  # [batch_size, 2 * num_components]

        # Reshape parameters to separate phi and psi
        batch_size = x.size(0)
        weights_logits = weights_logits.view(batch_size, 2, self.num_components)
        means = means.view(batch_size, 2, self.num_components)
        # Convert means to proper angle range (-180 to 180)
        means = torch.tanh(means) * 180.0
        # Ensure kappas are positive using softplus
        kappas = torch.nn.functional.softplus(kappas_raw.view(batch_size, 2, self.num_components))
        return weights_logits, means, kappas

    def forward(self, x, edge_index):
        """
        Forward pass through the model.
        
        Args:
            x: Input node features [batch_size, num_nodes, hidden_dim]
            edge_index: Edge indices for the graph [2, num_edges]
        
        Returns:
            Tuple of 
            weights_logits: Logits for mixture weights [batch_size, 2, num_components]
            means: Mean angles in degrees [batch_size, 2, num_components]
            kappas: Concentration parameters [batch_size, 2, num_components]
            sasa_pred: Predicted SASA values [batch_size]
            flex_pred: Predicted flexibility values [batch_size]
        """
        x = self.item_embedding(x)
        for layer in self.body:
            x = layer(x, edge_index)
        
        weights_logits_von_mises, means_von_mises, kappas_von_mises = self.predict_von_mises_parameters(
            x, self.head_von_mises, self.fc_weights_von_mises, self.fc_means_von_mises, self.fc_kappas_von_mises
        )

        values = self.head_values(x)
        # The multiplication with 2 is necessary because the last batch-norm (introduced for learning stability) seems to set off the predictions by factor 2
        sasa_pred = values[:, 0] * 2
        flex_pred = values[:, 1] * 2
        return weights_logits_von_mises, means_von_mises, kappas_von_mises, sasa_pred, flex_pred

    
def mixture_von_mises_nll(angles, weights_logits, mus, kappas):
    """
    Negative log-likelihood for mixture of von Mises distributions

    Args:
        angles: True angles in degrees [batch_size, 2] (phi, psi)
        weights_logits: Raw logits for mixture weights [batch_size, 2, n_components]
        mus: Mean angles in degrees [batch_size, 2, n_components]
        kappas: Concentration parameters [batch_size, 2, n_components]
    
    Returns:
        Negative log-likelihood
    """
    # Convert angles to radians
    angles_rad = angles * (np.pi / 180.0)
    mus_rad = mus * (np.pi / 180.0)

    # Normalize weights along component dimension
    weights = torch.nn.functional.softmax(weights_logits, dim=2)
    total_log_probs = []

    # Compute for phi and psi separately
    for angle_idx in range(angles.size(1)):
        # Extract values for this angle (phi or psi)
        angle_rad = angles_rad[:, angle_idx].unsqueeze(1)  # [batch_size, 1]
        angle_mu = mus_rad[:, angle_idx, :]  # [batch_size, n_components]
        angle_kappas = kappas[:, angle_idx, :]  # [batch_size, n_components]
        angle_weights = weights[:, angle_idx, :]  # [batch_size, n_components]

        # Compute von Mises PDF for each component
        # Using the formula: exp(kappa * cos(x - mu)) / (2*pi*I0(kappa))
        # For numerical stability, we approximate log(I0(kappa))
        cos_term = torch.cos(angle_rad - angle_mu)  # [batch_size, n_components]
        log_bessel = torch.log(torch.exp(angle_kappas) / torch.sqrt(2 * np.pi * angle_kappas + 1e-10))
        log_von_mises = angle_kappas * cos_term - np.log(2 * np.pi) - log_bessel
    
        # Apply weights and compute mixture log probability using logsumexp for numerical stability
        weighted_log_probs = torch.log(angle_weights + 1e-10) + log_von_mises  # [batch_size, n_components]
        angle_log_prob = torch.logsumexp(weighted_log_probs, dim=1)  # [batch_size]
        total_log_probs.append(angle_log_prob)  # Sum log probabilities across angles
    
    # Return negative mean log-likelihood
    return -torch.mean(total_log_probs[0]), -torch.mean(total_log_probs[1])


def train_model(
    model: torch.nn.Module, # graph neural network for analyzing glycans
    dataloaders: dict[str, torch.utils.data.DataLoader], # dict with 'train' and 'val' loaders
    optimizer: torch.optim.Optimizer, # PyTorch optimizer, has to be SAM if mode != "regression"
    scheduler: torch.optim.lr_scheduler._LRScheduler, # PyTorch learning rate decay
    num_epochs: int = 25, # number of epochs for training
):
    blank_metrics = {k: [] for k in {"loss", "phi_loss", "psi_loss", "sasa_loss", "flex_loss"}}
    metrics = {"train": copy.deepcopy(blank_metrics), "val": copy.deepcopy(blank_metrics)}
    best_loss = float("inf")

    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print(model.body[0].nn[0].weight[0, 0].isnan() == True)
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_metrics = copy.deepcopy(blank_metrics)
            running_metrics["weights"] = []

            for data in dataloaders[phase]:
                # Get all relevant node attributes
                x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # First forward pass
                    # weights_logits_von_mises, mus_von_mises, kappas_von_mises, weights_logits_gaussian, mus_gaussian, sigmas_gaussian = model(x.to("cuda"), edge_index.to("cuda"))
                    weights_logits_von_mises, mus_von_mises, kappas_von_mises, sasa_pred, flex_pred = model(x.to("cuda"), edge_index.to("cuda"))
                    y = y.to("cuda")
                    mono_mask = y[:, 2] != 0  # Do based on SASA
                    von_mises_phi_loss, von_mises_psi_loss = mixture_von_mises_nll(y[~mono_mask, :2], weights_logits_von_mises[~mono_mask], mus_von_mises[~mono_mask], kappas_von_mises[~mono_mask])
                    sasa_loss = torch.sqrt(torch.nn.functional.mse_loss(sasa_pred[mono_mask], y[mono_mask, 2]))  #  ** (1/2)
                    flex_loss = torch.sqrt(torch.nn.functional.mse_loss(flex_pred[mono_mask], y[mono_mask, 3]))  #  ** (1/2)
                    loss = von_mises_phi_loss + von_mises_psi_loss + sasa_loss / 60 + flex_loss
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Collecting relevant metrics
                running_metrics["loss"].append(loss.item())
                running_metrics["phi_loss"].append(von_mises_phi_loss.item())
                running_metrics["psi_loss"].append(von_mises_psi_loss.item())
                running_metrics["sasa_loss"].append(sasa_loss.item())
                running_metrics["flex_loss"].append(flex_loss.item())
                running_metrics["weights"].append(batch.max().cpu() + 1)

            # Averaging metrics at end of epoch
            for key in running_metrics:
                if key == "weights":
                    continue
                metrics[phase][key].append(np.average(running_metrics[key], weights = running_metrics["weights"]))

            print('{} Loss: {:.4f} Phi: {:.4f} Psi: {:.4f} SASA: {:.4f} Flex: {:.4f}'.format(
                phase, 
                metrics[phase]["loss"][-1], 
                metrics[phase]["phi_loss"][-1], 
                metrics[phase]["psi_loss"][-1], 
                metrics[phase]["sasa_loss"][-1], 
                metrics[phase]["flex_loss"][-1],
            ))

            # Keep best model state_dict
            if phase == "val":
                if metrics[phase]["loss"][-1] <= best_loss:
                    best_loss = metrics[phase]["loss"][-1]

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(metrics[phase]["loss"][-1])
                else:
                    scheduler.step()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    return metrics
