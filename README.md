# Fine-Grained Fruit Classification

A comprehensive deep learning project for fine-grained classification of fruit varieties using convolutional neural networks and later deployed on a robotic system (ROS).

Overview

The challenge of fine-grained classification remain in visual similarity and intra-class variability. Unlike coarse-grained classification (e.g., apple vs. banana), fine-grained classification requires models to learn subtle visual differences such as color gradients, texture patterns, and shape variations. For example, what are the differences between Jazz and Kanzi apples. Truthfully, it is a difficult task for a system and even a human to differentiate Jazz and Kanzi apples in different lighting environments.  

This project proposes an architecture inspired by a research paper *"Vision Transformers in Fine-Grained Visual Classification via Internal
Ensemble Learning Transformer"* by Q.Xu, J.Wang, and B.Luo 2023 to tackle such a challenge, where the goal is to distinguish between closely related cultivars within the same fruit species.

The project follows a phased approach, progressing from dataset collection and baseline model evaluation through fine-tuning and finally to real-world deployment on a robotic platform.
