# Movie Review Sentiment Analysis

This project implements a comprehensive sentiment analysis pipeline on movie reviews using various Natural Language Processing (NLP) techniques and machine learning models. The aim is to classify movie reviews as either positive or negative, providing insights into the effectiveness of different models and feature extraction methods in sentiment classification tasks.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models and Evaluation](#models-and-evaluation)
- [Results](#results)

## Introduction
Sentiment analysis is a crucial task in NLP that involves determining the emotional tone behind a body of text. In this project, we focus on the binary classification of movie reviews, categorizing them as either positive or negative. We explore several techniques for text preprocessing, vectorization, and classification using machine learning algorithms. The main goal is to compare different models and feature extraction methods to find the most effective approach for sentiment classification.

## Dataset
The project uses the [IMDb Movie Reviews](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) dataset. This dataset consists of 50,000 movie reviews split into 25,000 for training and 25,000 for testing. Each review is labeled as either positive (1) or negative (0). The dataset provides a rich source of textual data for training and evaluating sentiment analysis models.

## Project Workflow
1. **Load the Data**:
   - Download and extract the IMDb movie reviews dataset.
2. **Data Preprocessing**:
   - **HTML Tag Removal**: Clean the text data by removing HTML tags.
   - **Tokenization and Lemmatization**: Tokenize the text into words and apply lemmatization to normalize them.
   - **Stopwords Removal**: Remove common stopwords, keeping 'not' to retain sentiment context.
3. **Data Visualization**:
   - Generate word clouds to visualize the most frequent positive and negative words in the dataset.
4. **Data Splitting**:
   - Split the dataset into training (70%) and testing (30%) sets to evaluate model performance.
5. **Feature Extraction**:
   - **CountVectorizer**: Convert text into a matrix of token counts.
   - **TF-IDF**: Transform text into term frequency-inverse document frequency (TF-IDF) features.
6. **Model Training and Evaluation**:
   - Train multiple machine learning models (Logistic Regression, Random Forest, SVM, DNN) on the extracted features.
7. **Evaluation**:
   - Evaluate models using metrics such as accuracy, confusion matrix, F1 score, recall, and precision.

## Models and Evaluation
The following models were implemented and evaluated in this project:

- **Logistic Regression**:
  - Trained using both Bag-of-Words (BOW) and TF-IDF vectorized features.
- **Random Forest**:
  - Trained using both BOW and TF-IDF features.
- **Support Vector Machine (SVM)**:
  - Trained using TF-IDF features for better performance.
- **Deep Neural Network (DNN)**:
  - A neural network model trained using TF-IDF features to explore the potential of deep learning in sentiment analysis.

Each model was evaluated using accuracy, confusion matrix, F1 score, recall, and precision to identify the most effective method for sentiment classification.

## Results
The performance of each model is summarized in the table below:

| Vectorizer | Model                | Accuracy |
|------------|----------------------|----------|
| BOW        | Logistic Regression  | 88.92%   |
| TF-IDF     | Logistic Regression  | 89%      |
| BOW        | Random Forest        | 82.2%    |
| TF-IDF     | Random Forest        | 82.5%    |
| TF-IDF     | SVM                  | 89.5%    |
| TF-IDF     | DNN                  | 88.9%    |

From the results, the Support Vector Machine (SVM) with TF-IDF features showed the highest accuracy of 89.5%, indicating its effectiveness in handling the sentiment analysis task on this dataset. The Deep Neural Network (DNN) also performed competitively, demonstrating the potential of deep learning in text classification problems.
