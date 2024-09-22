# SpaRG: Sparsely Reconstructed Graphs for Generalizable fMRI Analysis

Welcome to the repository for **SpaRG**, a cutting-edge framework that utilizes sparsely reconstructed graphs to enhance fMRI analysis by optimizing for **sparsification** and **reconstruction**. This repository contains the code, data, and results for our work on improving the interpretability and robustness of functional connectome analyses.

## 📚 Background

Resting-state fMRIs (rs-fMRIs) offer rich insights into psychiatric disorders and personal traits. However, analyzing functional connectomes—measured via BOLD signal correlations—is challenging due to their high dimensionality and susceptibility to variations.

🧠 **SpaRG** addresses these challenges by leveraging **Graph Neural Networks (GNNs)** and **Variational Autoencoders (VAE)** to sparsify inputs during training, ultimately improving the classification accuracy while dramatically reducing the number of neural connections considered.

## 🔬 Key Features

- **Sparsification:** Learn a sparse mask during training that identifies robust neural connections while discarding irrelevant ones.
- **Reconstruction:** Use a Variational Autoencoder (VAE) to reconstruct the connectome from the sparse input, allowing better interpretability.
- **Classification:** Feed the reconstructed connectomes into a GNN for the final classification task.
- **Generalizability:** Achieves strong results across both in-distribution (ID) and out-of-distribution (OOD) data sites.

## 📊 Results

SpaRG achieves up to **99% reduction in neural connections** while maintaining or improving classification accuracy across both ID and OOD datasets.

|                | DiFuMo 64x64 | DiFuMo 1024x1024 |
|----------------|--------------|------------------|
| GCN        | 76.17±2.2    | 77.24±2.7        |
| FCN        | 73.94±4.2    | 78.34±2.7        |
| xGW-GAT    | 46.89±8.2    | 40.12±3.5        |
| Mask-GCN   | 76.14±2.6    | 76.83±3.4        |
| LASSO      | 82.10±8.4    | 83.74±2.4        |
| ElasticNet | 76.97±2.8    | 83.55±2.1        |
| Frobenius  | 74.24±5.9    | 82.55±5.0        |
| **SpaRG (ours)** | **82.40±4.5**  | **84.28±5.5**        |


## 🛠️ Installation

To get started with SpaRG, follow these simple steps:

1. Clone the repository

2. Navigate to the project directory:
   ```bash
   cd SpaRG
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the [ABIDE dataset](https://fcon_1000.projects.nitrc.org/indi/abide/) (or your chosen dataset).

## 🚀 Usage

To train the model and perform classification, use the following command (along with other chosen arguments):

```bash
python3 main.py --model_name sparg --sparse_method vae
```

## 🧑‍🔬 Contributors

- **Camila González** (Stanford University)
- **Yanis Miraoui** (Stanford University)
- **Yiran Fan** (Stanford University)
- **Ehsan Adeli** (Stanford University)
- **Kilian M. Pohl** (Stanford University)

This work was partly funded by the U.S. National Institute of Health (NIH), the Stanford HAI Google Cloud Credit, and the DGIST Joint Research Project.

## 📜 Citation

If you use this code, please cite our paper:

```
@inproceedings{2024sparg,
  title={Sparsely Reconstructed Graphs for Generalizable fMRI Analysis},
  author={Camila González and Yanis Miraoui and Yiran Fan and Ehsan Adeli and Kilian M. Pohl},
  booktitle={MLCN 2024},
  year={2024}
}
```

## 📬 Contact

For any questions or collaboration opportunities, please reach out to us.
