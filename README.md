<h1 align="center"> Lyric-Emotion-Classifier-with-BERT </h1>
<h2 align="center">Final project for the Technion's EE Deep Learning course (046211)</h2>
<h3 align="center"><small>Nir Voloshin & Yuri Minin</small></h3>


## Table of Contents
1. [Introduction](#introduction)
2. [Goals](#goals)
3. [Our Model](#our-model)
   3.1 [Architecture Overview](#architecture-overview)
   3.2 [Fine Tuning](#fine-tuning)
5. [Dataset](#dataset) 
6. [Experimental Setup](#experimental-setup)
7. [Results](#results)
8. [Conclusions](#conclusions)
9. [Future Work](#future-work)
10. [How To Run](#how-to-run)


## Introduction
BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model pre-trained on large corpus. 
It captures deep bidirectional context-aware representations for tokens in a sentence and is widely used as a base model for transfer learning in natural language processing (NLP).

BERT is typically fine-tuned for downstream tasks by adding task-specific layers on top of its pre-trained architecture. Depending on the nature of the task (classification, regression, sequence labeling), different outputs of the BERT model are used.

In this project, we employ BERT for sentences-level emotion regression, where the input is a chunk of song lyrics and the output is a continuous-valued emotion score between 0-1 (e.g., valence). 

By leveraging pretrained language models such as BERT, we seek to improve the understanding of emotional nuance in musical language. This approach has the potential to enhance music recommendation systems and at the end, our work opens the door to multimodal extensions that integrate audio, metadata, and listener feedback for richer emotion modeling.


## Goals
* Achieving well performance for our model.
* Improving performance compared to basic neural network.

## Our Model
Our model is based on the BERT-base architecture. The model takes as input a tokenized segment (chunk) of lyrics and outputs a continuous valence score representing the emotional content of the text.

### Architecture Overview:
* Backbone: bert-base-uncased pretrained language model (12 layers, 768 hidden size)
* Input: Full lyrics are segmented into overlapping text chunks to respect BERT’s maximum sequence length of 512 tokens.
   - Each chunk  is generated with a fixed stride to maintain context continuity.
   - Each chunk is processed independently by BERT.
   - For each original lyric (song), the final valence prediction is obtained by averaging the predictions from all chunks        belonging to the same lyric.

* Regression Head: A two-layer feedforward network:
  Linear(768 → 128) → ReLU → Linear(128 → 1)

### Fine Tuning:
To efficiently fine-tune BERT on our dataset, we incorporate Low-Rank Adaptation (LoRA), a parameter-efficient transfer learning method. LoRA is used to inject trainable low-rank matrices into linear layers (e.g., in self-attention).
In our setup, LoRA is applied to the query and value projection layers of BERT's self-attention modules in the final 4 layers (layers 9–12).

##Dataset


##Experimental Setup


##Results



##Conclusions



##Future Work



##How To Run
#Prerequisites







