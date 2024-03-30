# FakeNewsDetection-LSTM

# Project Title

Brief description of your project.

## Overview

Provide an overview of what the project does and its objectives.

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- tensorflow
- nltk

## Functionality

### Data Loading and Preprocessing

- **Pandas:** 
  - Used to read the CSV file containing the data.
  - Used for data manipulation and analysis.

- **Numpy:**
  - Utilized for numerical operations and handling arrays.
  
### Data Exploration

- **Data.head(5):** 
  - Displays the first 5 rows of the dataset.
  
- **Data.isnull().sum():** 
  - Calculates the sum of null values in each column of the dataset.
  
- **Data.shape:**
  - Returns the shape of the dataset (number of rows and columns).
  
- **sns.heatmap(Data1, yticklabels=False, cmap='magma'):**
  - Plots a heatmap to visualize the distribution of null values in the dataset. Setting `yticklabels=False` removes the y-axis labels for better visualization. The colormap `magma` is used to represent the intensity of null values, with darker shades indicating a higher concentration of null values.

### Data Cleaning

- **Data.dropna():**
  - Drops rows with missing values from the dataset.
  
- **Data.isnull().any():**
  - Checks if there are any null values left in the dataset after cleaning.

### Model Building

- **TensorFlow:**
  - Utilized for building deep learning models.
  
- **Sequential:**
  - Initializes a sequential model.
  
- **LSTM:**
  - Long Short-Term Memory layer used for sequential data processing.
  
- **Embedding:**
  - Converts words into dense vectors of fixed size.
  
- **Bidirectional:**
  - Creates a bidirectional wrapper for an RNN layer.

### Model Training and Evaluation

- **model.compile():**
  - Configures the model for training.
  
- **model.fit():**
  - Trains the model on the training data.
  
- **model.predict():**
  - Generates predictions on the test data.
  
- **accuracy_score():**
  - Calculates the accuracy of the model.
  
- **confusion_matrix():**
  - Creates a confusion matrix to evaluate the performance of the model.

### Visualization

- **sns.heatmap():**
  - Plots a heatmap to visualize the confusion matrix.

### Additional Functionality

- **model.get_weights():**
  - Retrieves the weights of the neural network model.

## Usage

This project focuses on detecting fake news using the attributes "title" and "text" from the dataset. To use this project:
- Ensure that you have the necessary libraries installed, including pandas, numpy, matplotlib, seaborn, tensorflow, and nltk.
- Load the dataset containing the "title" and "text" attributes.
- Preprocess the data, including handling missing values and cleaning the text data.
- Build a machine learning model, such as an LSTM neural network, to classify news articles as fake or real based on their titles and text content.
- Train the model on a labeled dataset and evaluate its performance using metrics such as accuracy and confusion matrix.
- Visualize the results, including the confusion matrix, to gain insights into the model's performance.

