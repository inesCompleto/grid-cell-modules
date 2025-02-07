import numpy as np
from scipy import io
import data_analysis.spatial.preprocessing as pp
import data_analysis.dim_reduction as dim_red
import sys


# Retrieve variables from MATLAB
rate_maps = np.array(eval(sys.argv[1]))  # Convert input string to numpy array
grid_score = np.array(eval(sys.argv[2]))  # Convert input string to numpy array

grid_score = np.nan_to_num(grid_score, nan=0) # replace 'nan' entries with '0'


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
args = [6] # tsne
#args=[0.3] # umap
reduced_data = dim_red.dim_red_algorithm('tsne', args, zscored_matrix, plots=True) 


#-------- Clustering of grid cell modules -------#
# 4) Clustering
# It returns a dict with info about the clusters (neuron index, grid scores, etc.)
cluster_info = dim_red.clustering(reduced_data, autocorrelograms, grid_score, min_neighbors=3)


# 5) Identify grid cell modules
def grid_module_condition(cluster):
    return np.mean(cluster['grid_scores']) > 0.3

modules = dim_red.id_grid_modules(reduced_data, cluster_info, condition=grid_module_condition, plots=True)
print(modules)




#-------- Optional -------#
# -- Plot autocorrelograms & rate maps of cell in modules --#

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


