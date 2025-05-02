# Neural Network Initialization and Activation Functions

This repository explores the impact of different weight initialization techniques and activation functions in training deep neural networks. The main goal is to understand how different initialization methods, such as **Bad Initialization**, **Xavier Initialization**, and **He Initialization**, affect the performance of neural networks when combined with various activation functions like **ReLU** and **Tanh**.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [Experiments](#experiments)
  - [Bad Initialization + Tanh](#bad-initialization--tanh)
  - [Xavier Initialization + Tanh](#xavier-initialization--tanh)
  - [Bad Initialization + ReLU](#bad-initialization--relu)
  - [He Initialization + ReLU](#he-initialization--relu)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Project Overview

Neural networks are sensitive to both their weight initialization strategy and the choice of activation function. Poor initialization can lead to slow convergence, dead neurons, or exploding/vanishing gradients. In this project, we experiment with different weight initialization methods and activation functions to examine their effects on the training process.

We specifically compare the following combinations:

- **Bad Initialization + Tanh**
- **Xavier Initialization + Tanh**
- **Bad Initialization + ReLU**
- **He Initialization + ReLU**

By visualizing the mean and standard deviation at each layer as well as examining the activation distributions, we can assess how these combinations affect the network's learning dynamics.

## Setup Instructions

To run this code, ensure you have the following Python libraries installed:

- `numpy`
- `matplotlib`

You can install these dependencies using `pip`:

```bash
pip install numpy matplotlib
Once the dependencies are installed, you can run the code directly. It will execute experiments using the different initialization methods and activation functions, and output the corresponding plots.
```

## Experiments
## Bad Initialization + Tanh
This experiment uses a bad initialization method where the weights are initialized with small random values (multiplied by 0.01). This causes the network to have very small weights initially, which can lead to dead neurons when combined with the Tanh activation function.

**Code Snippet:**
```bash
import numpy as np

def bad_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * 0.01  # Bad initialization

def tanh(x):
    return np.tanh(x)  # Tanh activation function
```

## Xavier Initialization + Tanh
Xavier initialization, also known as Glorot initialization, is used in this experiment. It initializes weights according to a distribution with variance 1 / n_in, where n_in is the number of input units to the layer. This helps maintain a reasonable variance in the activations across layers when combined with Tanh.

**Code Snippet:**
```bash
import numpy as np

def xavier_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(1. / fan_in)  # Xavier initialization

def tanh(x):
    return np.tanh(x)  # Tanh activation function
```

## Bad Initialization + ReLU
This experiment uses a bad initialization strategy combined with the ReLU activation function. ReLU activation can suffer from the dead neuron problem, especially when combined with poor weight initialization. This experiment will help visualize how this combination affects the network's activations.

**Code Snippet:**
```bash
import numpy as np

def bad_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * 0.01  # Bad initialization

def relu(x):
    return np.maximum(0, x)  # ReLU activation function
```

## He Initialization + ReLU
He initialization is designed for ReLU activations and scales the weights according to sqrt(2 / n_in), where n_in is the number of input units. This initialization method helps mitigate the vanishing gradient problem and accelerates training for ReLU-based networks.

**Code Snippet:**
```bash
import numpy as np

def he_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(2. / fan_in)  # He initialization

def relu(x):
    return np.maximum(0, x)  # ReLU activation function
```

## Results
The following visualizations are generated for each experiment:

**1. Layer-wise Mean Values:** This shows how the mean values of the activations change across layers for each initialization and activation function combination.

**2. Layer-wise Standard Deviation Values:** This shows the variance in activations, which can help identify issues such as exploding or vanishing gradients.

**3. Activation Distributions:** Histograms of activations at each layer, allowing us to visualize the effect of initialization and activation function on the distribution of values across the network layers.

## Conclusion
The experiments clearly demonstrate the importance of selecting an appropriate initialization method based on the activation function. For example, using ReLU with He initialization ensures proper variance propagation and prevents the vanishing gradient problem. In contrast, Bad Initialization methods combined with ReLU or Tanh can result in poor training behavior, with issues like dead neurons or slow convergence.

We conclude that careful weight initialization plays a crucial role in ensuring effective training for deep neural networks. By leveraging appropriate initialization techniques like Xavier and He, we can significantly improve the performance and stability of the training process.

## Usage
To run the experiments and visualize the results, follow these steps:
1. Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/initialization-method-experiment-with-NN.git
cd initialization-method-experiment-with-NN
```
2. Install the required dependencies:
```bash
pip install numpy matplotlib
```
3. Run the script that contains the experiments. For example, you can run a Python script like this:
```bash
python experiment.py
```
4. The script will generate plots of the layer-wise mean, standard deviation values, and activation distributions for each experiment. These plots will help you understand the effects of different weight initializations and activation functions on the training process.

