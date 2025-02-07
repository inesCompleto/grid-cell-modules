import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import os
import data_analysis.spatial.preprocessing as pp
import data_analysis.dim_reduction as dim_red
from sklearn.decomposition import PCA


#-------- Import data from .mat file -------#
def load_data(data_dir = "../data/", file="TestData_highGrid_trial1.mat"): 
    """
    Loads data from a .mat file in data_dir
    File contains info about: cellID, rat 2D trajectory, spatial information, gridness
    
    Args:
        - data_dir: pathway to file directory
        - file: name of file to import
    returns:
        - mat_file (dict)
    """
    mat_file = io.loadmat(data_dir + file)
    return mat_file

data = load_data()

print(data.keys()) # print entries of data dictionary

# Assign relevant variables
is_grid = data['isGrid']
rate_maps = data['rateMap']
grid_score = data['gridness']
grid_score = np.nan_to_num(grid_score, nan=0) # replace 'nan' entries with '0'
ell_gridness = data['gridness_ell']
ell_gridness = np.nan_to_num(ell_gridness, nan=0)  # replace 'nan' entries with '0'


# change rate_maps to have a shape that is convenient for the functions used in 'dimensionality reduction'
i=0
rate_maps_all = []
for i in range(len(grid_score)):
    eff_rate_map = np.nan_to_num(rate_maps[i][0], nan=0) # replace nan entries with zero
    rate_maps_all.append(eff_rate_map) # to go around weird shaping of rate_maps and ignore 1st dimension

has_nan = np.isnan(rate_maps_all).any() # optional: checks there still are any 'nan' entries


#-------- Dimensionlity reduction -------#
# 1) Autocorrelogram
n_total = np.shape(rate_maps_all)[0]

rate_map_shape = rate_maps_all[0].shape[0]*2-1
autocorrelograms = np.zeros((n_total, rate_map_shape, rate_map_shape)) #(number_neurons, shape of rate map for 1 neuron)

for i in range(n_total):
    autocorrelograms[i] = pp.autocorrelogram_2D(rate_maps_all[i])
   
    
# 2) Z-scored matrix of autocorrelograms
zscored_matrix = pp.zscore_spatial_autocorr(autocorrelograms)


# 3) Implement neighbour embedding algorihtm
# It returns a 2D projection of the high-dimensional data
# Options: tsne or umap. args = perplexity for tsne and min_dist for umap
#args = [6] # tsne
args=[0.3] # umap
reduced_data = dim_red.dim_red_algorithm('umap', args, zscored_matrix, plots=True) 


#-------- Clustering of grid cell modules -------#
# 4) Clustering
# It returns a dict with info about the clusters (neuron index, grid scores, etc.)
cluster_info = dim_red.clustering(reduced_data, autocorrelograms, grid_score,ell_gridness, min_neighbors=3)


# 5) Identify grid cell modules
def grid_module_condition(cluster):
    return np.mean(cluster['grid_scores']) > 0.3

modules = dim_red.id_grid_modules(reduced_data, cluster_info, condition=grid_module_condition, plots=True)
print(modules)




#-------- Optional -------#

# Option 1) Check autocorrelograms & rate maps of cell in modules

#n_modules = max(modules.keys())
#grid_mod_id = 1

#while grid_mod_id <= n_modules:
#    indices = modules[grid_mod_id]['index']
#    selected_neurons = [autocorrelograms[j] for j in indices]
#    mean_module = np.mean(selected_neurons, axis=0)    # Compute mean autocorrelogram of modules 

#    folder_name = f'Mod{grid_mod_id}' # Creates folder to save plots
#    os.makedirs(folder_name, exist_ok=True)
    
#    plt.figure()
#    plt.title(f'Mean module {grid_mod_id}', fontsize=20)
#    plt.imshow(mean_module)
#    plt.savefig(os.path.join(folder_name, f'Mean_mod{grid_mod_id}.png'), format='png')

#    for j in indices:
#        plt.figure()
#        plt.title(f'Cluster {grid_mod_id}, neuron {j}', fontsize=20)
#        plt.imshow(autocorrelograms[j])
#        plt.savefig(os.path.join(folder_name, f'autocorr_neuron_{j}.png'), format='png')
        
#        plt.figure()
#        plt.title(f'Cluster {grid_mod_id}, neuron {j}', fontsize=20)
#        plt.imshow(rate_maps_all[j])
#        plt.savefig(os.path.join(folder_name, f'rate_map_neuron_{j}.png'), format='png')
        
#    grid_mod_id+=1



 
#    mean_scales = np.mean(modules[grid_mod_id]['grid_scales'])
#    std_scales = np.std(modules[grid_mod_id]['grid_scales'])
#    print(f'Mean scales: {mean_scales}; std scales: {std_scales}')
    
#    mean_orientation = np.mean(modules[grid_mod_id]['orientation'])
#    std_orientation = np.std(modules[grid_mod_id]['orientation'])
#    print(f'Mean orientation: {mean_orientation}; std orientation: {std_orientation}')
     
#    mean_ell_gridness = np.mean(modules[grid_mod_id]['grid_ell'])
#    std_ell_gridness = np.std(modules[grid_mod_id]['grid_ell'])
#    print(f'Mean elliptic gridness: {mean_ell_gridness}; std elliptic gridness: {std_ell_gridness}')


# plot reduced data with only_in_traditional neurons highlighted
#colors = ['red' if i in only_in_traditional else 'white' for i in range(len(reduced_data))]
#plt.figure(figsize=(4, 3))
#plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=30, alpha=0.8, edgecolors="black", linewidths=0.5)
#plt.title('Only in traditional')
#plt.xlabel('dim 1')
#plt.ylabel('dim 2')
#plt.show()


# plot reduced data with only_in_tsne neurons highlighted
#colors = ['red' if i in only_in_tsne else 'white' for i in range(len(reduced_data))]
#plt.figure(figsize=(4, 3))
#plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=30, alpha=0.8, edgecolors="black", linewidths=0.5)
#plt.title('Only in umap')
#plt.xlabel('dim 1')
#plt.ylabel('dim 2')
#plt.show()



