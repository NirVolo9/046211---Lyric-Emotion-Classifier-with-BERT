<h1 align="center">🎵 <u>Lyric-Emotion-Classifier-with-BERT</u> 🎵</h1>

<h2 align="center">
  <i>Final project for the Technion's EE Deep Learning course (046211)</i>
</h2>

<h3 align="center">
  <sub><strong>Nir Voloshin</strong> & <strong>Yuri Minin</strong></sub>
</h3>

<p align="center">
<img src="assets/logo_1.png" alt="Lyric Emotion Classifier Logo" width="300"/>
</p>

## Table of Contents
1. [Introduction](#introduction)
2. [Goals](#goals)
3. [Our Model](#our-model)
   1. [Architecture Overview](#architecture-overview)
   2. [Fine Tuning](#fine-tuning)
5. [Dataset](#dataset) 
6. [Experimental Setup](#experimental-setup)
8. [Results](#results)
9. [Conclusions](#conclusions)
10. [EXTRA - Evaluation of Labeling and Training Strategies](#extra---evaluation-of-labeling-and-training-strategies)
11. [Future Work](#future-work)
12. [How To Run](#how-to-run)  
    1. [Added Files](#added-files)  
13. [References](#references)  

## Introduction
BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model pre-trained on large corpus. 
It captures deep bidirectional context-aware representations for tokens in a sentence and is widely used as a base model for transfer learning in natural language processing (NLP).

<p align="center">
  <img src="assets/bert.png" alt="BERT Google Logo" width="300"/>
</p>

BERT is typically fine-tuned for downstream tasks by adding task-specific layers on top of its pre-trained architecture. Depending on the nature of the task (classification, regression, sequence labeling), different outputs of the BERT model are used.

In this project, we employ BERT for sentences-level emotion regression, where the input is a chunk of song lyrics and the output is a continuous-valued emotion score between 0-1 (e.g., valence). 

By leveraging pretrained language models such as BERT, we seek to improve the understanding of emotional nuance in musical language. This approach has the potential to enhance music recommendation systems and at the end, our work opens the door to multimodal extensions that integrate audio, metadata, and listener feedback for richer emotion modeling.


## Goals
<p align="center">
  <img src="assets/goal.png" alt="Goal Target" width="250"/>
</p>

* Achieving well performance for our model.
  
* Improving performance compared to basic neural network.

## Our Model
Our model is based on the BERT-base architecture. The model takes as input a tokenized segment (chunk) of lyrics and outputs a continuous valence score representing the emotional content of the text.

### Architecture Overview:
* Backbone: bert-base-uncased pretrained language model (12 layers, 768 hidden size)
* Input: Full lyrics are segmented into overlapping text chunks to respect BERT’s maximum sequence length of 512 tokens. The Distribution of chunks per sample is shown in the histogram below. Some samples were even split into 11 chunks.
   - Each chunk  is generated with a fixed stride to maintain context continuity.
   - Each chunk is processed independently by BERT.
   - For each original lyric (song), the final valence prediction is obtained by averaging the predictions from all chunks        belonging to the same lyric.
* Regression Head: A two-layer feedforward network:
  Linear(768 → 128) → ReLU → Linear(128 → 1)
  
<img width="589" height="455" alt="chunks" src="https://github.com/user-attachments/assets/8bed137b-1207-4d6c-9ac8-2817139e8f3e" />



### Fine Tuning:
To efficiently fine-tune BERT on our dataset, we incorporate Low-Rank Adaptation (LoRA), a parameter-efficient transfer learning method. LoRA is used to inject trainable low-rank matrices into linear layers (e.g., in self-attention).
In our setup, LoRA is applied to the query and value projection layers of BERT's self-attention modules in the final 4 layers (layers 9–12).

### Model Flowchart
<img width="970" height="434" alt="flowchart" src="https://github.com/user-attachments/assets/8aa381ac-4640-4961-b479-8366258bd5f6" />

## Dataset
### 150K Lyrics Labeled with Spotify Valence  → Dataset `labeled_lyrics_cleaned.csv`
* Source: [Kaggle - Valence-labeled Lyrics](https://www.kaggle.com/datasets/edenbd/150k-lyrics-labeled-with-spotify-valence/)
* Details: The dataset consists of approximately 150,000 song lyrics from a wide range of artists. Each entry includes:
  artist name, full lyrics, song title, and valence score provided by Spotify representing the emotional positivity of the track.

   
## Experimental Setup
The model wasn't trained on the entire dataset due to hardware limitations. We took 30,000 samples out of the dataset.
The dataset the model was trained on :

* Training Dataset Size: 24,000 samples
* Validation\Test Dataset Size: 3,000 samples
* Batch Size: 32
* Learning Rate: 2e-5
* Num of epochs: 20

## Results
To evaluate the effectiveness of our approach, we compared our fine-tuned model to a baseline regressor that uses the same BERT-generated token embeddings as input to a simple MLP network.

<img width="968" height="266" alt="mlp_flowchart" src="https://github.com/user-attachments/assets/b5c8d1c9-165a-4c46-bcc0-1aa8f6162e1b" />

### The results for the two methods are as follows:

Baseline (MLP) Loss plots
<img width="790" height="490" alt="baseline_result" src="https://github.com/user-attachments/assets/57d02d9e-fe3b-44c1-8369-519b6c3c2faa" />

Fine-tuned BERT Loss plots
<img width="738" height="586" alt="model_result" src="https://github.com/user-attachments/assets/ecddad1b-3bf6-481e-a9a4-ea5ec7738c77" />

|Method      | MSE on test set|
|-------------|---------|
|baseline|   0.0517|
|proposed method|   0.0486|


## Conclusions
1. Fine-tuned BERT model with LoRA adaptation achieved slightly better performance compared to the baseline MLP model.
2. The overall prediction performance was not as strong as expected.
3. Exploring alternative language models, such as RoBERTa, may yield improved results due to their stronger pretraining and representational capacity. However, utilizing such models was not feasible within our available computational resources.  
4. Our suggested model struggled to fully capture the complexity of emotional expression in lyrics, indicating that valence prediction from text alone may be inherently limited — or that further architectural enhancements and richer multi-modal features (e.g., audio) may be needed.
5.  Future improvements could involve richer multi-modal representations (e.g., incorporating audio features), as well as more advanced architectures or alternative language models like RoBERTa which may be feasible given access to proper computational resources.

<p align="center">
  <img src="assets/thinking.png" alt="Robot thinking" width="250"/>
</p>

## EXTRA - Evaluation of Labeling and Training Strategies
Initially, we explored alternative datasets that might be more naturally suited to classification-based emotion prediction. Our goal was to find a dataset that could support a supervised mood classification task, and sufficient sample sizes for effective fine-tuning of a language model. However, none of the available datasets, apart from the one used in our project, satisfied both criteria- limiting our ability to pursue classification in a meaningful way.

Secondly, we aimed to classify lyrics into discrete mood categories. Since the dataset provided valence as a continuous score, we applied quantization into four, then six classes. Both approaches yielded disappointing results, likely due to many samples falling near class boundaries. 
This highlighted a fundamental mismatch between continuous emotional labels and discrete classification.
So, after several failed attempts and a few confused histograms, we metaphorically threw classification into the trash 🗑️.  
Therefore, we decided to proceed with a regression-based approach.

Moreover, we evaluated the impact of applying LoRA to different numbers of final layers in the BERT model, treating the number of affected layers as a hyperparameter. The goal was to determine how the depth of adaptation influences performance, with a focus on the last layers where task-specific information is typically concentrated.  
However, our experiments showed that varying the number of LoRA-adapted layers had only a marginal effect on overall performance, suggesting that this approach alone was not sufficient to significantly improve regression accuracy.


## Future Work
As mentions previously, future work could explore incorporating multi-modal inputs, such as audio features or acoustic embeddings, alongside lyrics. These additional modalities may help the model better capture the emotional nuances that are difficult to infer from text alone. 
However, pursuing this direction would require a suitable dataset that includes both lyrics and corresponding audio, along with reliable valence annotations.

## How To Run
1. Start by downloading labeled_lyrics_cleaned.csv from  
 [Kaggle - Valence-labeled Lyrics](https://www.kaggle.com/datasets/edenbd/150k-lyrics-labeled-with-spotify-valence/)
 place the file inside a directory named **data** .
2. Run the script data_preprocessing.py to clean and filter the dataset and saves he cleaned English-only version as english_lyrics_new.csv.
3. Place the file also inside  **data**.
4. Run Lyric_Emotion_Classifier.ipynb which will generate and save tokenized datasets to data/tokenized_dataset_/ with splits: train/ , val/ , test/
   Once created, you don't need to regenerate these for future runs unless the data changes so skip thses parts.
5. Run baseline_frozenBert_mlp.ipynb. It uses the saved tokenized dataset to train a shallow MLP on top of frozen BERT features.
6. Now you can compare results.

### Added Files
1. Lyric_Emotion_Classifier.ipynb 
2. data_preprocessing.py 
3. baseline_frozenBert_mlp.ipynb





<p align="center">
  <img src="assets/checklist.png" alt="Run Instructions Icon" width="280"/>
</p>


## References

* [150K Lyrics Labeled with Spotify Valence](https://www.kaggle.com/datasets/edenbd/150k-lyrics-labeled-with-spotify-valence/)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)





