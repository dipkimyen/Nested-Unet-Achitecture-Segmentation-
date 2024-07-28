# Nested U-Net Architecture for Segmentation

This repository contains a project for medical image segmentation using a Nested U-Net architecture. The project leverages a dataset of medical images to build a model capable of accurately segmenting regions of interest.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Medical image segmentation is a critical task in medical image analysis, enabling precise delineation of anatomical structures. This project implements a Nested U-Net architecture to improve segmentation accuracy over the traditional U-Net.

## Dataset

The dataset used in this project is the [KvasirCapsule SEG](https://datasets.simula.no/kvasir-capsule-seg/), a publicly available dataset for capsule endoscopy segmentation. The dataset contains images of polyps with their segmentation ground truth and bounding box information, annotated by an expert gastroenterologist.

### Dataset Details

KvasirCapsule-SEG is an enhanced subset of Kvasir-Capsule which includes:
- Polyp images
- Segmentation ground truth
- Bounding box information

Examples of polyps and their corresponding masks can be found in the dataset documentation.

### Suggested Evaluation Metrics

For evaluation purposes, we suggest using the following standard computer vision metrics:
- Dice Coefficient (DSC)
- Mean Intersection over Union (mIoU)
- Precision
- Recall
- Specificity
- Accuracy
- FPS

A detailed description of these metrics can be found in the referenced paper.

### Download

The dataset can be downloaded from [here](https://datasets.simula.no/kvasir-capsule-seg/).

### Citation

If you use this dataset in your research, please cite the following paper:

@inproceedings{jha2021nanonet,
title = {Nanonet: Real-time polyp segmentation in
video capsule endoscopy and colonoscopy},
author = {
Jha, Debesh and Tomar, Nikhil Kumar and Ali, Sharib and
Riegler, Michael A and Johansen, H{\aa}vard D and Johansen, Dag and
de Lange, Thomas and Halvorsen, P{\aa}l
},
booktitle = {Proceedings of the 2021 IEEE 34th International
Symposium on Computer-Based Medical Systems (CBMS)},
pages = {37--43},
year = {2021}
}


### Terms of Use

The use of the KvasirCapsule-SEG dataset is restricted for research and educational purposes. Commercial use is forbidden without prior written permission. For other purposes, contact the dataset creators.

### Ethics Approval

The dataset includes fully anonymized data approved by the Privacy Data Protection Authority and is exempted from approval from the Regional Committee for Medical and Health Research Ethics - South East Norway. All experiments were performed in accordance with relevant guidelines and regulations.

### Contact

Email debesh (at) simula (dot) no for questions about the dataset and research activities. Collaboration and joint research are always welcome!

## Installation

To run this project, you need to have Python and the following libraries installed:

- tensorflow
- keras
- numpy
- opencv-python
- matplotlib
- scikit-learn
- jupyter

You can install the required libraries using the following command:

'''bash
pip install tensorflow keras numpy opencv-python matplotlib scikit-learn jupyter'''

## Usage
Clone this repository:
'''bash
git clone https://github.com/dipkimyen/Nested-Unet-Achitecture-Segmentation.git'''
Navigate to the project directory:
'''bash
cd Nested-Unet-Achitecture-Segmentation'''
Open the Jupyter Notebook:
'''bash
jupyter notebook'''
Open the Nested_Unet_Segmentation.ipynb notebook and run the cells to see the data preprocessing, model training, and evaluation.

## Model Architecture
The Nested U-Net architecture extends the traditional U-Net by incorporating nested and dense skip connections to capture multi-scale features more effectively. This architecture enhances the model's capability to segment complex structures.
![image](https://github.com/user-attachments/assets/d5225a74-63b2-42eb-80cf-2ea65126522f)

## Key Features of Nested U-Net
- Nested and Dense Skip Connections: Unlike the traditional U-Net, the Nested U-Net introduces additional skip connections that create nested pathways for feature propagation. These dense connections help in capturing multi-scale features, allowing the model to learn more detailed representations.
- Multi-scale Feature Extraction: The architecture is designed to extract features at multiple scales, which is crucial for accurate segmentation of complex structures in medical images.
- Improved Gradient Flow: The dense skip connections also facilitate better gradient flow during training, addressing the vanishing gradient problem and leading to more stable training.

## Results
The results of the model's performance are summarized in the notebook. The performance metrics include accuracy, IoU (Intersection over Union), and Dice coefficient, which demonstrate the model's effectiveness in segmenting medical images.
![image](https://github.com/user-attachments/assets/797c6ee4-a310-4a86-a3b5-9d708b4f03df)

## Segmentation Results
Loss and Validation Curve
 ![image](https://github.com/user-attachments/assets/9df96382-b0ab-45d8-a4a0-29e2b35caf77)
 
## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
