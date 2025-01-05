# GenAI Potentials to Enhance Medical Image Classification

Paper ([Link](https://doi.org/10.48550/arXiv.2412.12532)):  
Addressing Small and Imbalanced Medical Image Datasets Using Generative Models: A Comparative Study of DDPM and PGGANs with Random and Greedy K Sampling 

**Table of Contents**   
[1. Abstract](#1-abstract)   
[2. Contributions](#2-contributions)   
[3. Contents](#3-contents)   
[4. Project Structure](#4-project-structure)   
[5. Setup](#setup)   
[6. Methodology](#methodology)   
[7. Dataset](#dataset)   
[8. Running the Code](#running-the-code)   
[9. Results](#results)   
[10. Contacts](#contacts)   
[11. Cite us](#cite-us)   

## 1. Abstract

This project addresses the challenges of small and imbalanced medical image datasets by exploring two generative models: Denoising Diffusion Probabilistic Models (DDPM) and Progressive Growing Generative Adversarial Networks (PGGANs). These models are used to generate synthetic images to augment medical datasets, which improves the performance of classification models.

We evaluate the impact of DDPM- and PGGAN-generated synthetic images on the performance of custom CNN, untrained VGG16, pretrained VGG16, and pretrained ResNet50 models, demonstrating significant improvements in model robustness and accuracy, especially in imbalanced scenarios.

For more details, please refer to the [paper](https://doi.org/10.48550/arXiv.2412.12532).

## 2. Our Contributions

- **An Evaluation Framework**: A comprehensive framework to systematically evaluate and compare the quality of images produced by DDPM and PGGANs.
- **High-Quality Image Generation**: Demonstrates that producing high-quality and diverse synthetic images using small medical image datasets is feasible.
- **Accuracy Improvement**: Incorporating synthetic images into the training datasets significantly improves the accuracy of classification models.
- **Increased Robustness**: Adding synthetic images to the original datasets enhances the robustness of classification models.
- **Faster Convergence**: The inclusion of synthetic images accelerates the convergence of classification models.

## 4. Contents of this repo

```sh
.
├── Code
│   ├── Classification Models
│   │   ├── Classification_128input
│   │   └── Classification_224input
│   ├── DDPM
│   │   └── DDPM_Pytorch.ipynb
│   ├── FID
│   │   ├── Results.txt
│   │   ├── fid.sh
│   │   ├── fid_comparison_plot.png
│   │   ├── fid_comparison_plot_full.png
│   │   └── fid_plot.ipynb
│   └── PGGANs
│       ├── ModelTrainingImages
│       ├── progan_modules.py
│       ├── train.py
│       ├── train_config_NRM200k_2024-04-11_20_17.txt
│       ├── train_config_PNM200k_2024-04-11_21_23.txt
│       ├── train_log_NRM200k_2024-04-11_20_17.txt
│       └── train_log_PNM200k_2024-04-11_21_23.txt
├── Dataset
│   ├── All_Data
│   │   ├── NORMAL
│   │   └── PNEUMONIA
│   ├── Generated_Images
│   │   ├── DDPM
│   │   ├── PGGANs
│   │   └── cGANs
│   ├── Mixed_Data
│   │   ├── Mixed150
│   │   └── PGGANs
│   └── Train
│       ├── NORMAL
│       └── PNEUMONIA
├── Figures
│   ├── DDPM_forward.png
│   ├── FID.png
│   ├── Normal_vs_Original_ddpm_3images.png
│   ├── Pneumonia_Original_ddpm_gans_3images.png
│   └── VGG16_and_CNN_performance_5 runs_2.png
├── requirements.txt
├── environment.yml
├── README.md
├── LICENSE

```

## Setup

### Prerequisites

- Python 3.x
- Conda or virtualenv

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/medical-image-classification.git
    cd medical-image-classification
    ```

2. Install the required packages using `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

3. Alternatively, create a conda environment using `environment.yml`:
    ```bash
    conda env create -f environment.yml
    conda activate your-environment-name
    ```

## Methodology
![alt text](Figures/Flowchart2.png)

## Dataset

The dataset used in this study consists of Chest X-ray (CXR) images with two classes: NORMAL and PNEUMONIA. The dataset is structured as follows:
- `dataset/NORMAL`: Contains normal CXR images.
- `dataset/PNEUMONIA`: Contains pneumonia CXR images.


![alt text](Figures/Dataset.png)


## Running the Code

### Training the VGG16 Model

1. Prepare the dataset:
    ```python
    from VGG_help import prepare_dataset
    dataset_dir = 'path/to/dataset'
    class_labels = ['NORMAL', 'PNEUMONIA']
    X, y = prepare_dataset(dataset_dir, class_labels)
    ```

2. Train the VGG16 model using cross-validation:
    ```python
    from VGG_help import cv_train_vgg_model
    fold_metrics_df, best_model = cv_train_vgg_model(X, y)
    ```

3. Plot training history:
    ```python
    from VGG_help import plot_train_history
    plot_train_history(fold_metrics_df, 'VGG16 Training History', 'vgg16_training_history.png')
    ```

### Training the Custom CNN Model

1. Load the dataset and prepare it as shown in the VGG16 training section.

2. Train the custom CNN model using cross-validation:
    ```python
    from CNN_Classification import fit_classification_model_cv
    fold_metrics_df, best_model = fit_classification_model_cv(X, y)
    ```

### Training DDPM

1. Open the `DDPM_Pytorch.ipynb` notebook.
2. Follow the instructions to train and evaluate the DDPM model.

### Training PGGANs

1. Train the PGGAN model using the `train.py` script:
    ```bash
    python train.py --path path/to/dataset --trial_name trial1 --gpu_id 0
    ```

<p align="center">
  <img src="Figures/Normal_gallary.png" alt="Normal vs Original DDPM" style="width:45%; margin-right: 5%;">
  <img src="Figures/Pneumina_gallary.png" alt="Pneumonia Original DDPM GANS" style="width:45%;">
</p>


### Calculating FID Scores

1. Open the `fid_plot.ipynb` notebook.
2. Follow the instructions to calculate and plot the FID scores.

![alt text](Figures/FID (1).png)

## Results

The results from the cross-validation and test set evaluations will provide insights into the performance improvements achieved by using synthetic images generated by DDPM and PGGANs.

![alt text](<Figures/Classification_boxplots.png>)
![alt text](<Figures/Classification_boxplots_F1.png>)

## Cite us 

- For any questions or issues, feel free to email Imran Khazrak (ikhazra@bgsu.edu) and/or Mostafa Rezaee (mostam@bgsu.edu).

- Also, please consider cite us as follows:

    - **IEEE style**:   
I. Khazrak, S. Takhirova, M. M. Rezaee, M. Yadollahi, R. C. Green II, and S. Niu, "Addressing Small and Imbalanced Medical Image Datasets Using Generative Models: A Comparative Study of DDPM and PGGANs with Random and Greedy K Sampling," arXiv preprint, vol. 2412.12532, 2024. [Online]. Available: https://arxiv.org/abs/2412.12532.

    - **BibTeX**:
        ```bibtex
        @misc{khazrak2024addressingsmallimbalancedmedical,
            title={Addressing Small and Imbalanced Medical Image Datasets Using Generative Models: A Comparative Study of DDPM and PGGANs with Random and Greedy K Sampling}, 
            author={Iman Khazrak and Shakhnoza Takhirova and Mostafa M. Rezaee and Mehrdad Yadollahi and Robert C. Green II and Shuteng Niu},
            year={2024},
            eprint={2412.12532},
            archivePrefix={arXiv},
            primaryClass={cs.CV},
            url={https://arxiv.org/abs/2412.12532}, 
        }
        ```
