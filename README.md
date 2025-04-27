# Fake News Detection Project

## Overview

This project aims to detect fake news using machine learning techniques. It utilizes a dataset of news articles labeled as either "real" or "fake" to train a model that can classify new, unseen articles.

## Libraries Used

-   `numpy`: For numerical computations.
-   `pandas`: For data manipulation and analysis.
-   `matplotlib.pyplot`: For creating visualizations.
-   `seaborn`: For enhanced data visualization.
-   `itertools`: For creating iterators for efficient looping.
-   `sklearn.model_selection.train_test_split`: To split the dataset into training and testing sets.
-   `sklearn.feature_extraction.text.TfidfVectorizer`: To convert text data into numerical features using the TF-IDF method.
-   `sklearn.linear_model.PassiveAggressiveClassifier`: A linear classifier suitable for large-scale learning.
-   `sklearn.metrics`: For evaluating model performance (accuracy, classification report, confusion matrix).

## Data

The dataset consists of news articles with the following columns:

-   `Unnamed: 0`: Index or ID (likely to be dropped)
-   `title`: Title of the news article.
-   `text`: Content of the news article.
-   `label`: Label indicating whether the news is "REAL" or "FAKE".

## Data Exploration

The dataset contains 6335 news articles. The "label" column has two unique values: "REAL" and "FAKE."

## Model

The model used is a `PassiveAggressiveClassifier`. This is a linear classifier particularly well-suited for large-scale datasets.

## Feature Extraction

The text data (article titles and content) is converted into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique.

## Performance

-   **Accuracy:** 93.92%
-   **Confusion Matrix:**
    ```
    [[591  37]
     [ 40 599]]
    ```

    *   True Negatives (TN): 591
    *   False Positives (FP): 37
    *   False Negatives (FN): 40
    *   True Positives (TP): 599


## Usage

1.  **Clone the repository:**

    ```
    git clone https://github.com/BrijeshRakhasiya/Fake-News-Detection.git
    ```

2.  **Install the required libraries:**

    ```
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

3.  **Run the Jupyter Notebook:**

    ```
    jupyter notebook Detecting-Fake-News.ipynb
    ```

4.  **Follow the code cells in the notebook:**
    *   Load the data.
    *   Preprocess the text data using TF-IDF.
    *   Split the data into training and testing sets.
    *   Train the `PassiveAggressiveClassifier`.
    *   Evaluate the model using accuracy, classification report, and confusion matrix.

## Next Steps

-   Evaluate the model's performance (accuracy, precision, recall, F1-score) on the test set.
-   Further data cleaning
-   Experiment with different machine-learning models (e.g., Naive Bayes, Random Forest) to improve performance.
-   Incorporate more advanced NLP techniques, such as word embeddings or transformer models, for better feature extraction.

## Author

**Brijesh Rakhasiya**
