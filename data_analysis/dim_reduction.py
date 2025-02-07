"""
Reduce dimensionality of data and identify clusters
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib.lines import Line2D
from kneed import KneeLocator
from sklearn.decomposition import PCA

def dim_red_algorithm(method, args, data, plots=True):
    """

    Args:
    - method (string): dimensionality reduction method to use (tsne, umap or pca)
    - data (ndarray): multi-dimensional data to analyze
    - plots (boolean): default True. plots the 2D transformed data 

    Returns:
    - reduced_data (ndarray): 2d transformed data
    """

    if method=='tsne': 
        model = TSNE(n_components=2, metric='euclidean', perplexity=args[0], early_exaggeration=10, random_state=0)

    elif method=='umap':
        model = umap.UMAP(n_components=2, metric='euclidean', n_neighbors=3, min_dist=args[0], init='pca', n_epochs = 500,
                               random_state=42)
    elif method=='pca':
        model =  PCA(n_components=2)
    else:
        print('Should one of the following methods: pca, umap or tsne')
        
    reduced_data = model.fit_transform(data.T)  

    if plots==True:
        plt.figure(figsize=(4, 4))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=50, color='grey', alpha=0.8)
        plt.title(f'2D {method} Projection') 
        plt.xlabel('tSNE Dimension 1')
        plt.ylabel('tSNE Dimension 2')
        plt.show()

    return reduced_data

def clustering(reduced_data, autocorrelograms, grid_score, min_neighbors):

    """
    Args:
    - reduced_data (ndarray):
    - autocorrelograms (ndarray):
    - grid_score (array):
    - min_neighbors (int):

    Returns:
    - cluster_info (dict): contains information of identified clusters. It is of the form
                            cluster_info = {
                                            0: {'index': [...],
                                                'grid_scores': [...]},
                                            1: { ...
                                            }
                            }
    """

    # Choose hyperparameters (distance epsilon)
    neigh = NearestNeighbors(n_neighbors=min_neighbors)
    neigh.fit(reduced_data)
    distances, _ = neigh.kneighbors(reduced_data)
    distances = np.sort(distances[:, min_neighbors-1])

    # Identify index of elbow
    x = range(len(distances))
    kneedle = KneeLocator(x, distances, S=0.5, curve='convex', direction='increasing', interp_method='interp1d')
    elbow_index = kneedle.elbow
    epsilon = distances[elbow_index]


    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.scatter(elbow_index, epsilon, color='red', label=f'Elbow (ε={epsilon:.2f})')
    plt.xlabel('Points')
    plt.ylabel(f'{min_neighbors}-th nearest neighbor distance')
    plt.title('K-distance Graph')
    plt.show()

    # Identify clusters
    dbscan = DBSCAN(eps=epsilon, min_samples= min_neighbors)
    labels = dbscan.fit_predict(reduced_data)
    n_clusters = np.max(labels) + 1

    # Store cluster information (grid score of all neurons in each cluster) in a dict:
    # cluster_label: [grid scores]
    i = 0
    cluster_info = {}  # Initialize with an empty list for entry 0

    while i < n_clusters:
        neuron_index = []
        scores = []
        for j, label in enumerate(labels):
            if label == i:
                scores.append(grid_score[j][0])
                neuron_index.append(j)
         
            cluster_info[i] = {"index": [],
                               "grid_scores": []}
    
            # Add the neuron data to the corresponding lists
            cluster_info[i]["index"].extend(neuron_index)
            cluster_info[i]["grid_scores"].extend(scores)
        
        i+=1

    return cluster_info



def id_grid_modules(reduced_data, cluster_info, condition, plots=True):

    """
    
    Returns:
    - modules (dict): contains index of clusters identified as being grid cell modules. 
                      Module '0' contains the id of all the neurons that do not belong to a grid module. It has the form
                      modules = {0: [...],
                                1: [...]}
    """


    n_clusters = max(cluster_info.keys()) + 1

    i=0
    m=1
    modules = {0:{"index": [],
                  "grid_scores": []}}

   # modules = {0:[]}

    while i < n_clusters:
        if condition(cluster_info[i]):
            modules[m] = {"index": [],
                          "grid_scores": []}
    
            # Add the neuron data to the corresponding lists
            modules[m]["index"].extend(cluster_info[i]['index'])
            modules[m]["grid_scores"].extend(cluster_info[i]['grid_scores'])

        #    modules[m] = cluster_info[i]['index']
            m+=1
        else:
            modules[0]["index"].extend(cluster_info[i]['index'])
            modules[0]["grid_scores"].extend(cluster_info[i]['grid_scores'])

        #    modules[0].extend(cluster_info[i]['index'])
        i+=1

    if plots == True:

        # Define color map
        label_colors = {0: 'white', 1: '#CC7722', 2: '#85B09A', 3: '#505086', 
                4: '#A31621', 5: '#7A86C0', 6: '#E3B23C', 7: '#A3BCF9', 8: '#C3642F', 9: '#54615E'}

        cluster_labels = np.zeros(len(reduced_data), dtype=int) # Initialize an empty array to store cluster labels 
                                                                # for each data point

        #print(modules.items())
        for cluster, dict in modules.items():  # Fill in cluster_labels based on modules
         #   cluster_labels[dict] = cluster
            cluster_labels[dict['index']] = cluster

        colors = [label_colors[label] for label in cluster_labels]

        # Create a custom legend
        custom_legend = []
        for i in range(1,m):
            custom_legend.append(Line2D([0], [0], color=label_colors[i], lw=4))

        # Create figure
        plt.figure(figsize=(4, 3))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=30, alpha=0.8, edgecolors= "black", linewidths=0.5)
        plt.title('Grid cell modules')
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        plt.legend(custom_legend, [f'Module {i}' for i in range(1,m)],
                   loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

    return modules
