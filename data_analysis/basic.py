"""
Basic preprocessing of spiking neural data.
"""


import numpy as np

def firing_rates(n_neurons, spks, t_range, dt):
    """
    Compute firing rates in period t_range.
    
    Args:
        - n_neurons (int): number of neurons
        - spks (ndarray): spike trains of n_neurons
        - t_range (array): array [t_init, t_end]
        - dt (float): size of time bin 
    Returns:
        - fr (ndarray): firing rates of n_neurons
    """

    fr = np.zeros((n_neurons, len(t_range)-1))
    for i in range(n_neurons):
        cnt, _ = np.histogram(spks[i], t_range)
        fr[i] = cnt/dt

    return fr

