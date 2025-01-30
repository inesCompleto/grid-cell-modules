"""
Analyze spatial autocorrelogram to identify grid cells

-

"""

import numpy as np
import matplotlib.pyplot as plt 
import math
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters


def central_peak_radius(autocorrelogram, grid_peaks=False):
    
    """Find the radius of the central peak based on the radial profile.

    Args:
      - autocorrelogram (np.ndarray): 
      - grid_peaks (boolean): if True, estimate peak distances from central peak

    Returns:
      - minima[0] or distance_to_threshold (int): radius of central peak given by the indices of the radial_profile     
    """ 
    
    center = np.array(autocorrelogram.shape) // 2

    # 1) Compute radial profile (average autocorrelation value over concentric rings, for each radial distance bin)
    y, x = np.indices(autocorrelogram.shape) # generates grid indices for the shape of the autocorrelogram.
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2) # calculates radial distance from center
    r = r.astype(int) # converts distance to int, since distance needs to be given in terms of bins
    tbin = np.bincount(r.ravel(), autocorrelogram.ravel()) # flattens r and autocorrelogram, and sums up values of autocorrelogram that correspond to each value in r
    nr = np.bincount(r.ravel()) # ????
    radial_profile = tbin / nr # average value of autocorrelogram for each radial distance bin

    if grid_peaks == True:
        idx_peaks = find_peaks(radial_profile)[0]
        print('Indices of peaks in radial profile:', idx_peaks)

    # 2) First local minimum or the first drop below 0.2
    minima = find_peaks(-radial_profile)[0] # get indices where local minima in radial_profile occur
    distance_to_threshold = np.where(radial_profile < 0.2)[0][0] if len(np.where(radial_profile < 0.2)[0]) > 0 else len(radial_profile) # if no element in radial profile is <0.2, consider the len of the radial_profile array
                                                                                                                                        # this avoids evaluating an empty array in the next step and associated errors
    if len(minima) > 0 and minima[0] < distance_to_threshold:
        return minima[0]
    else:
        return distance_to_threshold



def compute_angle(v1, v2):
    norm1 = np.sqrt(np.sum(v1 ** 2))
    norm2 = np.sqrt(np.sum(v2 ** 2))
    return math.acos(np.sum(v1 * v2) / (norm1 * norm2))

def grid_scale(autocorrelogram, plot_peaks=False):
    """
    Compute grid scale and orientation.

    Parameters:
        - autocorrelogram (np.ndarray): spatial autocorrelogram
    Returns:
        - scale (float): grid scale 
        - orientation (float): orientation of grid
    """

    autocorrelogram_copy = np.copy(autocorrelogram)
   
    size = autocorrelogram_copy.shape[0]

    # Exclude center peak (set it to zero)
    center_x, center_y = size // 2, size // 2
    peak_radius = central_peak_radius(autocorrelogram_copy)
    autocorrelogram_copy[(center_x-peak_radius):(center_x+peak_radius), (center_y-peak_radius):(center_y+peak_radius)] = 0

    # Parameters
    neighborhood_size = 3*peak_radius
    threshold = np.mean(autocorrelogram_copy)  # Adjust based on data

    # Find local maxima
    data_max = filters.maximum_filter(autocorrelogram_copy, neighborhood_size)
    maxima = (autocorrelogram_copy == data_max)
    data_min = filters.minimum_filter(autocorrelogram_copy, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    # Label maxima and find coordinates
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2
        y.append(y_center)

    # Center coordinates for grid distance calculation
    center = np.array(autocorrelogram_copy.shape) / 2

    # Compute distances and retain the 6 closest peaks
    distances = np.sqrt((np.array(x) - center[1])**2 + (np.array(y) - center[0])**2)
    sorted_indices = np.argsort(distances)[:6]
    x_closest = np.array(x)[sorted_indices]
    y_closest = np.array(y)[sorted_indices]

    # Plotting if needed
    if plot_peaks==True:
        plt.figure()
        plt.imshow(autocorrelogram_copy, cmap='gray')
        plt.plot(x_closest, y_closest, 'ro')
        plt.show()

    # Compute distance of each peak to the center
    distance = np.sqrt((np.array(x_closest) - center[1])**2 + (np.array(y_closest) - center[0])**2)

    # Scale distance to normalize interval
    interval_length = 1.0 / ((size - 1) / 2)
    distance = distance * interval_length #  scaled distances of the peaks from the center.
    peaks = np.array([(x_closest - center_x)*interval_length, (y_closest - center_y) * interval_length])  # relative positions of the peaks, centered at the origin
    scale = np.median(distance)

    angle = []
    if len(peaks[1])==0:
        orientation = None
    else:
        for i in range(len(peaks[1])):
            peak_vector = np.array([peaks[0, i], peaks[1, i]])  # Get the x and y components of the i-th peak
            angle.append(compute_angle(peak_vector - center, np.array([-1, 0])))
        orientation = np.min(angle)


    return scale, orientation



def gridness(autocorrelogram):

    """
    -----
    This function computes gridness score accoring to [1]. The auto
    correlation of the firing rate map is rotated in 3 degree steps. The
    resulting gridness score is the difference between a minimum of cross
    correlations at 60 and 90 degrees, and a maximum of cross correlations at
    30, 90 and 150 degrees.

    The center of the auto correlation map (given by corr_cutRmin) is removed
    from the map.

    References
    ----------
    .. [1] Hafting, T. et al., 2005. Microstructure of a spatial map in the
       entorhinal cortex. Nature, 436(7052), pp.801-806.
    """

    size = autocorrelogram.shape[0]
    angles = [30, 60, 90, 120, 150] 

    # 1) Remove central peak
    center_x, center_y = size // 2, size // 2
    peak_radius = central_peak_radius(autocorrelogram)
    autocorrelogram[(center_x-peak_radius):(center_x+peak_radius), (center_y-peak_radius):(center_y+peak_radius)] = 0

    # 2) Rotate autocorrelogram
    corrs_og_rot = {} # store the correlation values between the og autocorrelogram and rotated versions.

    for angle in angles:
        rotated_autocorr = scipy.ndimage.rotate(autocorrelogram, angle, reshape=False) # rotate autocorrelogram
        corr, _ = pearsonr(autocorrelogram, rotated_autocorr) # calculate Pearson correlation

        corrs_og_rot[angle] = corr  

    gridness = (min(corrs_og_rot[60], corrs_og_rot[120]) - max(corrs_og_rot[30], corrs_og_rot[90], control_corrs[150]))

    return gridness

 
def shuffle_spikes(x, y, fr, num_shuffles=100):
    """Shuffling of spikes by shifting firing data."""
    shuffled_scores = []
    session_duration = len(x)
    
    for _ in range(num_shuffles):
        shift = np.random.randint(20, session_duration - 20)  # Shift by random amount
        fr_shuffled = np.roll(fr, shift)  # Time-shifting spike train
        rate_map_shuffled, _ = create_rate_map(x, y, fr_shuffled, bin_size=0.05)
        grid_score_shuffled = grid_score(rate_map_shuffled)
        shuffled_scores.append(grid_score_shuffled)
    
    return np.percentile(shuffled_scores, 95)