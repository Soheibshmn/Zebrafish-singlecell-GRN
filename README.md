#Uncovering Developmental Trajectories in Zebrafish Embryogenesis using scRNA-seq
This repository contains the Python code for a scientific report on analyzing single-cell RNA sequencing (scRNA-seq) data from zebrafish embryos. The primary goal of this project is to model the developmental trajectory of notochord cells, infer the underlying gene regulatory networks (GRNs), and identify key transcription factors (TFs) driving cell fate decisions during embryogenesis.

The pipeline leverages a combination of standard bioinformatics tools, machine learning models (Variational Autoencoders and Graph Neural Networks), and advanced trajectory analysis techniques to provide a comprehensive view of cellular development.

#Table of Contents
Methodology (https://www.google.com/search?q=%23-methodology)

Key Features (https://www.google.com/search?q=%23-key-features)

Data Sources (https://www.google.com/search?q=%23-data-sources)

System Requirements (https://www.google.com/search?q=%23-system-requirements)

Installation (https://www.google.com/search?q=%23-installation)

How to Run (https://www.google.com/search?q=%23-how-to-run)

Output Files (https://www.google.com/search?q=%23-output-files)

License (https://www.google.com/search?q=%23-license)

Citation (https://www.google.com/search?q=%23-citation)

Methodology
The analysis pipeline proceeds through the following key stages:

Data Preprocessing: Raw scRNA-seq count matrices from different time points are loaded, merged, and subjected to rigorous quality control (QC), filtering, normalization, and scaling.

Feature Engineering & Dimensionality Reduction: The data is initially reduced using PCA. A deep learning approach combining a Variational Autoencoder (VAE) and a Graph Attention Network (GAT) is then used to learn a robust low-dimensional latent space that captures complex cellular relationships.

Trajectory Inference & Pseudotime Analysis: Using the learned embeddings, cells are ordered along a developmental path (pseudotime). The pipeline employs multiple methods, including Diffusion Pseudotime (DPT) and Partition-based graph abstraction (PAGA), and includes a grid search to find parameters that maximize the correlation with real developmental stages.

Gene Regulatory Network (GRN) Inference: A GRN is constructed to model the regulatory relationships between transcription factors and their target genes. The analysis identifies key TFs based on network centrality and out-degree.

RNA Velocity & Advanced Modeling: RNA velocity is computed to infer the future state of individual cells. Additionally, the Kardar-Parisi-Zhang (KPZ) equation is applied to the trajectory to model its dynamic properties.

Visualization & Evaluation: The pipeline generates a wide range of visualizations, including 2D/3D UMAPs, force-directed graphs, GRN plots, and heatmaps of gene expression over pseudotime. Finally, a set of comprehensive metrics is calculated to evaluate the quality of the results.

Key Features
End-to-End Pipeline: From raw data loading to final visualization and evaluation.

Hybrid Dimensionality Reduction: Combines PCA with a VAE and a GNN (GAT) for enhanced feature representation.

Robust Pseudotime Analysis: Includes an enhanced grid search to find the optimal trajectory that correlates with known developmental stages.

GRN Inference: Identifies key transcriptional regulators and their targets.

Advanced Dynamics: Incorporates RNA velocity and KPZ equation modeling.

Comprehensive Visualization: Produces a rich set of static and interactive plots, including 3D trajectories and detailed network graphs.

Data Sources
scRNA-seq Data: The single-cell RNA sequencing data for zebrafish embryogenesis was obtained from the Gene Expression Omnibus (GEO) under accession number GSE112294 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE112294).

Transcription Factor List: The list of known zebrafish (Danio rerio) transcription factors was sourced from AnimalTFDB 4.0 (Danio_rerio_TF.txt (https://guolab.wchscu.cn/AnimalTFDB4/#/Download)).

System Requirements
Python 3.8+

Operating System: Linux, macOS, or Windows

RAM: 16 GB or more is recommended for handling the scRNA-seq datasets.

A CUDA-enabled GPU is recommended for accelerating VAE and GNN training but not required.

Installation
Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

Install the required Python packages. It is highly recommended to use a virtual environment.

# Create and activate a virtual environment (optional but recommended)
python -m venv
venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

# Install packages using pip
pip install numpy pandas matplotlib seaborn torch torch_geometric \
            scipy scikit-learn umap-learn scanpy anndata networkx \
            igraph scvelo plotly openpyxl

Note: torch_geometric may have specific installation requirements depending on your system and CUDA version. Please refer to the official PyG installation guide.

How to Run
Set up the data directory.

Create a main data directory (e.g., Zfish_Data).

Inside this directory, place the seven time-point CSV files (e.g., GSM3067189_04hpf.csv).

Place the metadata file (GSE112294_ClusterNames.csv) in the same directory.

Place the raw TF list (Danio_rerio_TF.txt) in the same directory.

 (https://www.google.com/search?q=%23-methodology)Update the scr (https://www.google.com/search?q=%23-key-features)ipt paths.

Op (https://www.google.com/search?q=%23-data-sources)en the script total c (https://www.google.com/search?q=%23-system-requirements)ode 2 best ver (https://www.google.com/search?q=%23-installation)sion (from a (https://www.google.com/search?q=%23-how-to-run)ll).py.

Modif (https://www.google.com/search?q=%23-output-files)y the DAT (https://www.google.com/search?q=%23-license)A_DIR vari (https://www.google.com/search?q=%23-citation)able to point to the absolute path of your data directory.

# Define paths
DATA_DIR = r"C:\path\to\your\Zfish_Data"

Execute the main pipeline script.

python "total code 2 best version (from all).py"

The script will run the full analysis pipeline, printing progress information to the console and saving output files to a results subdirectory within your DATA_DIR.

Running from a saved state:
After the first successful run, an adata_processed_enhanced.h5ad file is created. To skip the time-consuming preprocessing and model training steps on subsequent runs, you can set load_from_h5ad=True in the main() function call at the bottom of the script.

Output Files
The script will create a results folder inside your DATA_DIR containing:

AnnData Objects (.h5ad):

adata_processed_enhanced.h5ad: Data after preprocessing and model training.

adata_final_enhanced.h5ad: The final AnnData object with all analysis results.

Visualizations (.png, .html):

comprehensive_results.png: A multi-panel plot showing UMAPs, pseudotime correlation, top TFs, and more.

3d_trajectory_enhanced.html: An interactive 3D plot of the developmental trajectory.

grn_network.png / grn_detailed.png: Visualizations of the inferred gene regulatory network.

tf_dynamics_heatmap.png: Heatmap of TF expression over pseudotime.

rna_velocity_umap.png: UMAP with RNA velocity streams.

...and many other plots for different analysis steps.

Data Files (.csv, .txt):

final_metrics.csv: A summary of key evaluation metrics (e.g., pseudotime correlation, cluster count).

tf_expression_dynamics.csv: Smoothed expression values for TFs across pseudotime bins.

kpz_end_state.txt: The final state from the KPZ equation simulation.

License
This project is licensed under the MIT License. See the LICENSE file for details.
