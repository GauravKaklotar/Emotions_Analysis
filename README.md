# Emotion Detection from Text

This project focuses on classifying emotions in Twitter text data into six categories: sadness, joy, love, anger, fear, and surprise. The dataset consists of Twitter messages, each labeled with one of these emotions.

## Project Overview

1. **Data Preprocessing**:
   - Loaded the Twitter text dataset and performed text cleaning.
   - Removed URLs, mentions, hashtags, and special characters from the text.
   - Tokenized and normalized the text for consistent analysis.

2. **Exploratory Data Analysis (EDA)**:
   - Analyzed the distribution of emotions in the dataset.
   - Examined the length of text segments to understand the data structure.
   - Generated visualizations, including bar plots and histograms, to gain insights into the dataset.

3. **Model Selection and Training**:
   - Chose Logistic Regression for initial model training.
   - Vectorized the text data using TF-IDF to convert it into numerical features.
   - Split the dataset into training and testing sets with an 80/20 ratio.
   - Trained the model on the training data and evaluated its performance.

4. **Model Evaluation**:
   - Assessed the model using accuracy, precision, recall, and F1-score metrics.
   - Generated a classification report to evaluate the performance across different emotion categories.
   - Created a confusion matrix to visualize the model’s prediction accuracy.

5. **Model Improvement**:
   - Performed hyperparameter tuning using Grid Search to optimize the Logistic Regression model.
   - Validated the model using cross-validation to ensure it generalizes well to unseen data.
