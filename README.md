# Multimodal Fake News Detection

## Table of Contents
1. [Team](#team)
2. [Motivation](#motivation)
3. [Introduction](#introduction)
4. [Methodology](#methodology)
    - [NLP Task Addressed](#nlp-task-addressed)
    - [Dataset](#dataset)
    - [Preprocessing](#preprocessing)
    - [Models and Techniques](#models-and-techniques)
5. [Experiments and Results Evaluation](#experiments-and-results-evaluation)
    - [ERNIE Model - Surya Putrevu](#ernie-model---surya-putrevu)
    - [GPT Model - Rohan Singh](#gpt-model---rohan-singh)
    - [BERT Model - Kajal Patil](#bert-model---kajal-patil)
6. [Related Work](#related-work)
7. [Interesting Insights](#interesting-insights)
8. [Future Work](#future-work)
9. [Team Contribution](#team-contribution)
10. [Code Repository](#code-repository)
11. [Conclusions](#conclusions)
12. [Future Directions](#future-directions)
13. [References](#references)

## Team
- Surya Putrevu - [sputre2@pdx.edu](mailto:sputre2@pdx.edu)
- Kajal Patil - [kajal@pdx.edu](mailto:kajal@pdx.edu)
- Rohan Singh - [rohsingh@pdx.edu](mailto:rohsingh@pdx.edu)
- Dan Shitkar - [dshitkar@pdx.edu](mailto:dshitkar@pdx.edu)

## Methodology

### NLP Task Addressed
Binary classification of news based on title and images associated with it.

### Dataset
We utilized the ‘multimodalfake’ dataset by Hariharan RL, consisting of textual features and image URLs. Dataset size: approximately 59000 samples.

### Preprocessing
Limited dataset size to 10000 rows for faster experimentation. Selected relevant columns for analysis. Dataset was split into 80:20 ratio to form train and test data respectively.

### Models and Techniques
Implemented ERNIE, GPT, and BERT models along with image vectorization using the OPENCV library. Developed a neural network architecture to consume both text and image inputs.

## Experiments and Results Evaluation

### ERNIE Model - Surya Putrevu
Achieved 86% accuracy with significant precision differences for fake and real news.

### GPT Model - Rohan Singh
Focused on text, achieved 77% accuracy. Impact of discarding author and domain fields discussed.

### BERT Model - Kajal Patil
Experimented with different dataset sizes, achieving 82.56% accuracy for a dataset of 10,000 records.

## Related Work
Our approach takes inspiration from past research on multimodal models, enhancing it by combining Transformer models with image processing techniques.

## Interesting Insights
- Successful integration of text and image data.
- Challenges in handling diverse image sources and quality.
- Experimentation revealed the need for further data augmentation.

## Conclusions
Throughout our project, we conducted a thorough investigation into the effectiveness of a multimodal NLP model for text and image classification.

## Future Directions
Ongoing efforts include expanding the dataset, incorporating advanced image processing techniques, and exploring transfer learning for model improvement.
