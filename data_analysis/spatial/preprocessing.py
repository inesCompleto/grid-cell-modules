"""
Preprocessing of spiking neural data for spatial analysis.
"""


import numpy as np
from scipy.signal import correlate2d
from scipy.stats import zscore 


def spatial_rate_map(x, y, fr, bin_size=0.05):
    
    """
    Create a 2D rate map from position and firing rate data.
    
    Parameters:
    - x, y: Arrays of the rat's x and y positions over time.
    - fr: Array of firing rates of the neuron over time.
    - bin_size: The size of each bin in the 2D space.
    - x_range, y_range: Ranges of x and y for the grid. If None, they will be based on the data.
    
    Returns:
    - rate_map: 2D array where each bin contains the average firing rate in that bin.
    - occupancy_map: 2D array showing the occupancy time in each bin.
    """

    # Define the range of the map
    bins_x = np.arange(np.min(x), np.max(x) + bin_size, bin_size)
    bins_y = np.arange(np.min(y), np.max(y) + bin_size, bin_size)

    # Counting the time spent in each spatial bin by calculating how often the rat was in each bin (each time step can be considered a unit of time)
    # It counts how many times a data point (rat's position) falls into each spatial bin.
    occupancy_map, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y])

    # Create a map that sums firing rate in each bin.
    # Instead of counting the number of occurrences in each bin, you sum the neuron's firing rate in each bin
    firing_map, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y], weights=fr)

    # Average firing rate: firing_map / occupancy_map
    with np.errstate(divide='ignore', invalid='ignore'): # suppress 'divsion by zero' warnings
        rate_map = firing_map / occupancy_map
        rate_map[occupancy_map == 0] = np.nan  # Set bins with no occupancy to NaN
    
    return rate_map 


def autocorrelogram_2D(rate_map):
    """
    Compute spatial autocorrelogram
    """

    rate_map_centered = rate_map - np.nanmean(rate_map) # Subtract mean from the rate map, ignoring the nan entries
    rate_map_centered = np.nan_to_num(rate_map_centered) # Convert nan entries into zeros
    autocorrelogram = correlate2d(rate_map_centered, rate_map_centered, mode='full')
    
    # Normalize the autocorrelogram (optional)
    min_val = np.min(autocorrelogram)
    max_val = np.max(autocorrelogram)

    # Check for division by zero
    if max_val == min_val:
        autocorrelogram[:] = 0  # Set all entries to zero if there's a division by zero risk
    else:
        autocorrelogram = (autocorrelogram - min_val) / (max_val - min_val)
    
    return autocorrelogram


def zscore_spatial_autocorr(autocorrelograms):
    """
    Processes autocorrelograms by converting them into z-standardized column vectors
    and then concatenating them into a matrix.

    Parameters:
    autocorrelograms (np.ndarray): 3D array where each 2D slice along the first axis
                                   is the autocorrelogram of a neuron.

    Returns:
    result_matrix (np.ndarray): Matrix with spatial bins as rows and cells as columns.
    """
    
    # Get the number of neurons
    num_neurons = autocorrelograms.shape[0]

    # Initialize an empty list to store the column vectors
    column_vectors = []

    # Process each autocorrelogram
    for neuron in range(num_neurons):
        # Extract the 2D autocorrelogram for the current neuron
        ac = autocorrelograms[neuron]
        
        # Flatten the 2D autocorrelogram into a 1D column vector
        column_vector = ac.flatten()

        # Z-standardize the column vector
        z_standardized_vector = zscore(column_vector)

        # Append the z-standardized column vector to the list
        column_vectors.append(z_standardized_vector)
    
    # Concatenate all column vectors to form the matrix
    result_matrix = np.column_stack(column_vectors)

    # Check if any NaN values are present. If so, tSNE does not run.
    # This can happen if rate map is all zero, 
    #which in turn can happen when all the entries of rate map from Laurenz are nan
    has_nan = np.isnan(result_matrix).any()
    if has_nan==True:
        result_matrix[np.isnan(result_matrix)] = 0

    return result_matrix


