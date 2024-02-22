# GPT-FL: Generative Pre-Trained Model-Assisted Federated Learning

## Table of Contents
* Overview
* Installation
* Usage
* Contact

## Overview
In this work, we propose GPT-FL, a generative pre-trained model-assisted FL framework that effectively addresses the issues of existing methods. The key idea behind GPT-FL is to leverage the knowledge from the generative pre-trained models and to decouple synthetic data generation from the federated training process. 
Specifically, GPT-FL prompts the generative pre-trained models to generate diversified synthetic data. These generated data are used to train a downstream model on the server in the centralized manner, which is then fine-tuned with the private client data under the standard FL framework. 
By doing this, the proposed GPT-FL is able to combine the advantages of previous methods while addressing their limitations.

Paper Link: https://arxiv.org/abs/2306.02210

The GPT-FL package contains:
