
# Grid Cell Module Identification

## Overview

This repository contains a Python implementation for identifying and characterising grid-cell modules from neuronal spatial firing-rate maps.

Grid cells are neurons found in the medial entorhinal cortex that exhibit spatially periodic firing patterns, forming a hexagonal grid-like representation of the environment. Grid cells are often organised into modules, with neurons within a module sharing similar spatial scales and firing patterns.

The aim of this project is to identify groups of grid cells with similar spatial firing characteristics using spatial autocorrelograms, dimensionality reduction and clustering techniques.

The analysis pipeline is designed to explore the organisation of grid-cell populations and identify potential grid-cell modules in experimental neuronal recordings.

## Repository structure

```text
grid-cell-modules/
│
├── data/
│   └── ...                         # Input experimental data (.mat files)
│
├── data_analysis/
│   ├── spatial/
│   │   └── preprocessing.py        # Spatial data preprocessing
│   └── dim_reduction.py            # Dimensionality reduction and clustering
│
├── id_grid_modules.py              # Main analysis pipeline
│
└── README.md
```

## Methodology

The analysis consists of the following steps:

### 1. Data loading

Experimental data are loaded from MATLAB `.mat` files using `scipy.io.loadmat`.

The main variables used by the analysis are:

- `rateMap`: Spatial firing-rate maps for individual neurons.
- `gridness`: Gridness scores associated with each neuron.

Missing values (NaNs) in the firing-rate maps and gridness scores are replaced with zeros.

### 2. Spatial autocorrelograms

A two-dimensional spatial autocorrelogram is computed for each neuron's firing-rate map.

Autocorrelograms capture the spatial periodicity of neuronal firing patterns and provide a representation that can be used to compare grid-cell activity independently of the absolute spatial position of the firing fields.

### 3. Data standardisation

The resulting autocorrelograms are z-scored to standardise the data before dimensionality reduction.

### 4. Dimensionality reduction

Dimensionality reduction is performed to represent the high-dimensional autocorrelogram data in a two-dimensional space.

Two algorithms are supported by the dimensionality-reduction module:

- **t-SNE (t-distributed Stochastic Neighbour Embedding)**
- **UMAP (Uniform Manifold Approximation and Projection)**

The current implementation uses t-SNE with a perplexity parameter of 6.

UMAP can alternatively be selected by modifying the corresponding parameters in `id_grid_modules.py`.

### 5. Clustering

Neurons are clustered based on their representation in the reduced-dimensional space.

The clustering function uses the reduced data, spatial autocorrelograms and gridness scores to identify groups of neurons with similar spatial firing characteristics.

The current implementation uses a minimum-neighbour parameter of 3.

### 6. Grid-cell module identification

Candidate modules are identified from the resulting clusters.

A cluster is considered a grid-cell module when its mean gridness score is greater than 0.3:

```python
def grid_module_condition(cluster):
    return np.mean(cluster['grid_scores']) > 0.3
```

The resulting modules contain information about the neurons assigned to each module.

### 7. Results export

The indices of neurons belonging to each identified module are extracted and saved to a MATLAB `.mat` file:

`grid_modules_index.mat`

Each module is stored under a separate variable (e.g. `module_1`, `module_2`, etc.), containing the indices of its corresponding neurons.

## Installation

The project requires Python and the scientific Python ecosystem.

The main script directly uses:

- NumPy
- SciPy

Additional dependencies may be required by the dimensionality-reduction and preprocessing modules, depending on the selected implementation.

Clone the repository:

```bash
git clone https://github.com/inesCompleto/grid-cell-modules.git

cd grid-cell-modules
```

Install the main dependencies:

```bash
pip install numpy scipy
```

Install any additional dependencies required by the modules in `data_analysis/` before running the analysis.

## Usage

### 1. Prepare the data

Place the experimental MATLAB data file in the `data/` directory.

The default input file specified in the main script is:

`TestData_highGrid_trial1.mat`

The input file should contain at least the following variables:

- `rateMap`
- `gridness`

### 2. Configure the analysis

Open `id_grid_modules.py` and adjust the input filename if necessary:

```python
def load_data(data_dir="./data/", file="TestData_highGrid_trial1.mat"):
```

The dimensionality-reduction algorithm and its parameters can also be modified.

For t-SNE:

```python
args = [6]
reduced_data = dim_red.dim_red_algorithm(
    'tsne', args, zscored_matrix.T, plots=False
)
```

For UMAP:

```python
args = [0.3]
reduced_data = dim_red.dim_red_algorithm(
    'umap', args, zscored_matrix.T, plots=False
)
```

### 3. Run the analysis

From the repository root, execute:

```bash
python id_grid_modules.py
```

The identified modules and their corresponding neuron indices are printed to the console.

### 4. Retrieve the results

The analysis automatically saves the identified module indices in:

`grid_modules_index.mat`

This file can be loaded in Python using SciPy or in MATLAB for further analysis.

## Optional visualisation

The main script also contains commented-out code for visualising:

- Individual neuronal spatial firing-rate maps.
- Individual spatial autocorrelograms.
- Mean autocorrelograms for each identified module.

These visualisations can be enabled to inspect the spatial characteristics of the identified modules.
