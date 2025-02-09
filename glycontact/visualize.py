import seaborn as sns
import networkx as nx
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from io import BytesIO
from pathlib import Path
from scipy.cluster import hierarchy
from collections import Counter
from IPython.display import Image, display
from glycontact.process import inter_structure_variability_table, get_structure_graph, monosaccharide_preference_structure
from glycowork.motif.draw import GlycoDraw


def draw_contact_map(act, export='', size = 0.5, return_plot=False) :
        ### shows an atom_contact_map as heatmap using an atom_contact_table created by make_atom_contact_table()
        # act : the atom_contact_table, a dataframe generated by make_atom_contact_table()
        # export : name of the exported figure if any
        # size = text size
        sns.set(font_scale=size)
        heatmap = sns.heatmap(act,xticklabels=True,yticklabels=True)
        heatmap.set_yticklabels(heatmap.get_xticklabels(), rotation=0)
        if export:
                plt.savefig(export, bbox_inches='tight')
        if return_plot:
                return heatmap
        else:
                plt.show()


def make_gif(prefix, tables):
    ### Create a gif using multiple PNG files
    # prefix: prefix to include in the filename of the gif output
    # tables: result for each frame
    output_path = f'{prefix}_animation.gif'
    images = []
    for table in tables:
        ax = draw_contact_map(table, return_plot=True)
        fig = ax.figure
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        images.append(imageio.imread(buf))
        plt.close(fig)  # Clean up memory
    if images:
        imageio.mimsave(output_path, images, duration=0.2, loop=0)
        display(Image(filename=output_path))


def show_correlations(corr_df,font_size=1):
    ### Uses a correlation matrix as dataframe (corr_df) to represent it as a heatmap
    sns.set(font_scale=font_size)
    # Visualize the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Distances')
    plt.tight_layout()
    plt.show()


def show_correlation_dendrogram(corr_df, font_size = 1):
    ### Create a hierarchical clustering dendrogram from a correlation matrix as dataframe (corr_df)
    plt.figure(figsize=(10, 8))
    linkage = hierarchy.linkage(corr_df.values, method='ward')
    dendrogram = hierarchy.dendrogram(
        linkage, labels=corr_df.columns,
        leaf_rotation=90, leaf_font_size=8
    )
    # Group results by cluster
    monolist = dendrogram['ivl']
    clustlist = dendrogram['leaves_color_list']
    res_dict = {color: [] for color in set(clustlist)}
    for mono, clust in zip(monolist, clustlist):
        res_dict[clust].append(mono)
    plt.tick_params(axis='x', which='major', labelsize=font_size)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Residue')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()
    return res_dict


def plot_monosaccharide_instability(glycan, format='png', mode='sum'):
    # plot monolink variability for all clusters of a given glycan
    # possible formats: png, pdf
    # mode: sum, mean
    variability_table = inter_structure_variability_table(glycan)
    stability_scores = (variability_table.sum() if mode == 'sum' 
                       else variability_table.mean())
    sorted_scores = sorted(stability_scores.items(), key=lambda x: x[1])
    labels, values = zip(*sorted_scores)
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xlabel('Monosaccharides')
    plt.ylabel('Variability score')
    plt.xticks(rotation=90)
    plt.tight_layout()
    if format in ('png', 'pdf'):
        plt.savefig(f'{glycan}_instabilities.{format}')
    plt.show()


def plot_glycan_score(glycan, score_list=[], attribute="Weighted SASA", save_plot=False) :
    ### Displays a given glycan and highlights monosaccharides using a score list
    # score_list : list of raw values used to highlight monosaccharides (example: mean SASA score, standard deviation...)
    if not score_list:
        ggraph = get_structure_graph(glycan)
        scores = np.array(list(nx.get_node_attributes(ggraph, attribute).values()))
    else:
        scores = np.array(score_list[:-1])  # Remove -R value
    # Normalize scores
    score_range = scores.max() - scores.min()
    normalized_scores = (scores - scores.min()) / score_range if score_range > 0 else np.zeros_like(scores)
    filepath = f"{glycan}_highlighted.pdf" if save_plot else ''
    return GlycoDraw(glycan, per_residue=normalized_scores.tolist(), filepath=filepath)


def show_monosaccharide_preference_structure(df, monosaccharide, threshold, mode='default'):
  #df must be a monosaccharide distance table correctly reanotated
  #mode can be 'default' (check individual monosaccharides in glycan), 'monolink' (check monosaccharide-linkages in glycan), 'monosaccharide' (check monosaccharide types)
  res_dict = monosaccharide_preference_structure(df, monosaccharide, threshold, mode)
  value_counts = Counter(res_dict.values())
  # Plotting the histogram
  plt.bar(value_counts.keys(), value_counts.values())
  plt.xlabel('Values')
  plt.ylabel('Frequency')
  plt.title(f'Frequency of Encountered Values for {monosaccharide} above {threshold}')
  plt.tight_layout()
  plt.show()


def plot_superimposed_glycans(superposition_result, output_file=None, show_labels=True):
    """
    Create a 3D plot of superimposed glycan structures.
    
    Args:
        superposition_result: Output from superimpose_glycans()
        output_file: Optional path to save plot
        show_labels: Whether to show atom labels
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    # Plot reference structure
    ref_coords = superposition_result['ref_coords']
    ax.scatter(ref_coords[:, 0], ref_coords[:, 1], ref_coords[:, 2],
              c='blue', marker='o', s=100, label='Reference', alpha=0.7)
    # Plot transformed mobile structure
    transformed = superposition_result['transformed_coords']
    ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2],
              c='red', marker='o', s=100, label='Mobile (aligned)', alpha=0.7)
    # Add labels if requested
    if show_labels:
        # Plot reference labels
        for i, label in enumerate(superposition_result['ref_labels']):
            ax.text(ref_coords[i, 0], ref_coords[i, 1], ref_coords[i, 2],
                   label, fontsize=8, color='blue')
        # Plot mobile labels
        for i, label in enumerate(superposition_result['mobile_labels']):
            ax.text(transformed[i, 0], transformed[i, 1], transformed[i, 2],
                   label, fontsize=8, color='red')
    # Add RMSD to title
    rmsd = superposition_result['rmsd']
    ax.set_title(f'Superimposed Glycan Structures\nRMSD: {rmsd:.2f} Å')
    # Set labels and legend
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.legend()
    # Adjust view
    ax.view_init(elev=20, azim=45)
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
