# Spam Email Classification Using Machine Learning

## Overview

This project classifies emails as Spam or Non-Spam using the Spambase dataset. The notebook walks through the full workflow: loading data, cleaning, exploratory analysis, outlier handling, feature scaling, and model evaluation.

The final notebook now compares three classifiers on the clean preprocessed data:

- Support Vector Machine (SVM)
- Naive Bayes
- Decision Tree

It also shows a graphical comparison of all model scores and ends with a short conclusion about the best performer.

## Dataset

- Dataset: Spambase
- Task: Binary classification
- Samples: 4,601 emails
- Features: 57
- Target: `spam` where `0 = Non-Spam` and `1 = Spam`
- Missing values: None found in the dataset

### Feature Groups

The dataset features fall into three groups:

1. Word frequency features
2. Character frequency features
3. Capital run length features

## Notebook Flow

1. Import libraries
2. Load the dataset
3. Add column names
4. Inspect shape, info, and summary statistics
5. Check missing values and duplicates
6. Explore class distribution and correlations
7. Split the data into training and testing sets
8. Scale features with `StandardScaler`
9. Train and evaluate a baseline model before outlier removal
10. Detect outliers with the IQR method
11. Remove outliers from the training set only
12. Re-scale the cleaned data
13. Train and evaluate the cleaned-data models
14. Compare SVM, Naive Bayes, and Decision Tree results
15. Plot model performance and write the final conclusion

## Model Results

The notebook prints the following metrics for each model:

- Accuracy
- Precision
- Recall
- F1 Score

The current run on cleaned data showed that SVM performed best overall, with Naive Bayes showing strong recall and Decision Tree remaining competitive.

## Visual Output

The notebook generates these plots:

- Spam vs Non-Spam class distribution
- Correlation heatmap
- Feature boxplots
- Outlier visualizations before and after removal
- Confusion matrices
- Grouped bar chart comparing the final model scores

## Setup

### Requirements

- Python 3.7 or newer
- Jupyter Notebook or JupyterLab

### Install dependencies

```bash
pip install -r requirements.txt
```

### Open the notebook

```bash
jupyter notebook
```

Then open `Spam_Email_Classification.ipynb`.

## Project Structure

```text
Bootcamp/Day1/
├── Spam_Email_Classification.ipynb
├── README.md
├── requirements.txt
└── spambase Dataset/
    ├── spambase.data
    ├── spambase.names
    └── spambase.DOCUMENTATION
```

## Key Libraries

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Notes

- Run the notebook from top to bottom so each variable is created before it is reused.
- The cleaned-data comparison section is at the bottom of the notebook.
- The final comparison chart is a grouped bar chart that shows all four metrics for all three models.

## Conclusion

The main takeaway is that the cleaned-data pipeline is most useful when you want to compare multiple models on the same filtered dataset. In this project, SVM gives the strongest overall balance of Accuracy, Precision, Recall, and F1 Score.

## Reference

- UCI Spambase dataset: https://archive.ics.uci.edu/ml/datasets/spambase
