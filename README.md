📊 End-to-End Deep Learning Pipeline with Manual Neural Network from Scratch
============================================================================

This repository demonstrates a full pipeline of a deep learning classification project executed without relying on high-level libraries like `torch.nn`. The architecture, loss function, training loop, forward propagation, and backpropagation are written completely from scratch using `PyTorch` core functionalities, showcasing raw understanding and mathematical clarity in neural networks.

👉 **Colab Notebook Link**: [Open in Google Colab](https://colab.research.google.com/drive/1Enk9PC1ikD2g6EZvQdPJ5RsrVHFmtVIr#scrollTo=qjG0yHtjRG3q)

* * *

### 💻 Technical Stack
------------------

![Python](https://img.shields.io/badge/Python-3.8%252B-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)  
![NumPy](https://img.shields.io/badge/NumPy-1.24-yellow)  
![Pandas](https://img.shields.io/badge/Pandas-1.5-orange)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-green)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-blueviolet)  
![Google-collab](https://img.shields.io/badge/Jupyter-Notebook-orange)
    

* * *

### 🚀 Project Workflow & Core Components

#### 📁 Data Processing & EDA

*   Two separate datasets (`train.csv`, `test.csv`) were imported, verified for null values, duplicates, and outliers.
    
*   Detailed Exploratory Data Analysis (EDA) using correlation heatmaps, distribution plots, and boxplots helped extract trends and identify potential feature scales or inconsistencies.
    

#### 🧠 Custom Neural Network (From Scratch)

*   Implemented a manual neural network using core PyTorch tensors and operations — **no `nn.Module`, no autograd shortcuts**.
    
*   Each component is explicitly defined:
    
    *   Weight and bias initialization with gradient tracking
        
    *   Forward propagation using matrix operations
        
    *   Activation functions (sigmoid)
        
    *   Custom binary cross-entropy loss with clipping to avoid log instability
        
    *   Manual gradient descent updates with `.backward()` and `with torch.no_grad()`
        

#### 🔁 Training Loop

*   Manually structured epoch-based training loop.
    
*   Loss computed and backpropagated in every epoch.
    
*   Weight and bias updates carried out without using an optimizer class.
    
*   Gradients zeroed out explicitly to ensure clean update steps.
    

#### 📈 Evaluation

*   Predictions on test data followed by accuracy calculation using raw tensor comparison.
    
*   Applied sigmoid thresholding to simulate classification decision boundaries.
    

* * *

### 🧠 What This Demonstrates

This project reflects:

*   A **ground-up understanding of how deep learning models work internally**
    
*   The ability to implement **manual optimization and training logic**
    
*   Fluency with **data handling, visualization, and statistical insights**
    
*   **Hands-on skills in PyTorch**, not just model usage but framework internals
    
*   The drive to go beyond tutorials — building components instead of importing them