# Spam Email Classification Using Machine Learning

## Project Overview

This project focuses on classifying emails as **Spam** or **Non-Spam** using the **Spambase dataset**. The project demonstrates a complete machine-learning workflow, including data preprocessing, missing-value analysis, exploratory data analysis, visualization, outlier detection and removal, and classification.

**Main Objective:** Compare the performance of a Logistic Regression classification model **before and after outlier removal** to determine whether removing statistical outliers improves model performance.

---

## Dataset Description

**Dataset:** Spambase
- **Domain:** Computer Science / Email Classification
- **Task:** Binary Classification
- **Instances:** 4,601 email samples
- **Features:** 57 input features
- **Target Variable:** `spam` (0 = Non-Spam, 1 = Spam)
- **Feature Types:** Integer and Continuous values
- **Missing Values:** None (verified during analysis)

### Feature Categories

The dataset contains three types of features:

1. **Word Frequency Features (48 features)**
   - Frequency of common spam words (e.g., "free", "money", "remove", "internet")
   - Frequency of company names (e.g., "hp", "george")
   - Technical terms (e.g., "data", "lab", "telnet")

2. **Character Frequency Features (6 features)**
   - Frequency of special characters: `;` `(` `[` `!` `$` `#`

3. **Capital Letter Run Length Features (3 features)**
   - Average length of capital letter sequences
   - Longest capital letter sequence
   - Total number of capital letters

---

## Project Workflow

The notebook implements the following complete ML pipeline:

```
1. Data Loading
   ↓
2. Dataset Understanding & Info Display
   ↓
3. Missing Value & Duplicate Analysis
   ↓
4. Exploratory Data Analysis (EDA)
   ↓
5. Data Visualization
   ↓
6. Train-Test Split (80-20)
   ↓
7. Feature Scaling (StandardScaler)
   ↓
8. Classification Before Outlier Removal
   ↓
9. Model Evaluation & Metrics
   ↓
10. Outlier Detection (IQR Method)
    ↓
11. Outlier Removal
    ↓
12. Feature Re-scaling
    ↓
13. Classification After Outlier Removal
    ↓
14. Model Evaluation & Metrics
    ↓
15. Performance Comparison
    ↓
16. Final Conclusion & Analysis
```

---

## Installation & Setup

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- pip package manager

### Step 1: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import pandas, numpy, matplotlib, seaborn, sklearn; print('All libraries installed successfully!')"
```

### Step 3: Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open `Spam_Email_Classification.ipynb` in your browser.

---

## Project Structure

```
Bootcamp/Day1/
│
├── Spam_Email_Classification.ipynb    # Main notebook (40 steps)
│
├── requirements.txt                    # Python dependencies
│
├── README.md                           # This file
│
└── spambase Dataset/
    ├── spambase.data                   # Dataset file (4,601 samples)
    ├── spambase.names                  # Feature descriptions
    └── spambase.DOCUMENTATION          # Dataset documentation
```

---

## Key Sections of the Notebook

### Data Preprocessing (Steps 1-9)
- Import libraries
- Load dataset
- Add descriptive column names
- Understand dataset shape
- Display dataset info and statistics
- Analyze missing values
- Check for duplicates
- Analyze target variable distribution

### Exploratory Data Analysis (Steps 10-15)
- Visualize spam vs non-spam distribution
- Generate correlation heatmap
- Create boxplots for important features
- Identify potential outliers visually
- Prepare features and target variables

### Model Training Before Outlier Removal (Steps 16-23)
- Train-Test split (80-20 with stratification)
- Feature scaling using StandardScaler
- Train Logistic Regression classifier
- Make predictions
- Evaluate model (Accuracy, Precision, Recall, F1-Score)
- Generate confusion matrix

### Outlier Detection & Removal (Steps 24-30)
- Calculate quartiles (Q1, Q3) and IQR
- Detect outlier rows using IQR method
- Visualize outliers using boxplots
- Remove outliers from training data
- Calculate percentage of data removed
- Visualize data distribution after removal

### Model Training After Outlier Removal (Steps 31-39)
- Re-scale cleaned training data
- Train new Logistic Regression classifier
- Make predictions
- Evaluate model
- Generate confusion matrix
- Compare performance metrics
- Visualize performance improvements

### Final Analysis (Steps 40)
- Generate comprehensive summary
- Compare all metrics before vs after
- Calculate performance changes
- Provide conclusion and interpretation

---

## Model Details

### Algorithm
**Logistic Regression** - A linear classification algorithm suitable for binary classification tasks.

### Hyperparameters
- `max_iter=1000` - Maximum iterations for convergence
- `random_state=42` - For reproducibility

### Train-Test Split
- Training: 80% of data
- Testing: 20% of data
- Stratification: Maintains class distribution in both sets

### Feature Scaling
- Method: StandardScaler (Z-score normalization)
- Formula: `(X - mean) / std`
- Fitted on training data only (to prevent data leakage)

### Outlier Detection
- Method: Interquartile Range (IQR)
- Lower Bound: Q1 - 1.5 × IQR
- Upper Bound: Q3 + 1.5 × IQR
- Rows with any value outside bounds are marked as outliers

---

## Expected Results

The notebook will produce:

1. **Data Insights**
   - Dataset dimensions and structure
   - Class distribution
   - Missing values count

2. **Visualizations**
   - Count plots (spam distribution)
   - Correlation heatmap
   - Boxplots (features by class)
   - Boxplots (before and after outlier removal)
   - Confusion matrices (before and after)
   - Performance comparison bar chart
   - Improvement visualization

3. **Classification Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Classification Reports
   - Confusion Matrices

4. **Performance Comparison**
   - Before vs After metrics
   - Improvement/decrease in each metric
   - Detailed interpretation

---

## Interpretation of Results

### Accuracy
- The percentage of correct predictions out of total predictions
- **Before Outlier Removal:** Baseline performance
- **After Outlier Removal:** Performance on cleaned data

### Precision
- Of all spam predictions, what percentage were correct?
- Important when false positives are costly (e.g., marking legitimate emails as spam)

### Recall
- Of all actual spam emails, what percentage did the model identify?
- Important when false negatives are costly (e.g., missing spam emails)

### F1-Score
- Harmonic mean of Precision and Recall
- Useful when you need a single metric that balances both concerns

### Performance Improvement
- **Positive values:** Outlier removal improved the metric
- **Negative values:** Outlier removal decreased the metric
- **~0.00:** No significant change

---

## Possible Outcomes

### Scenario 1: Improvement After Outlier Removal
- Outliers were noisy/erroneous data
- Model generalizes better on cleaner data
- Recommend removing outliers

### Scenario 2: Decrease After Outlier Removal
- Outliers represent legitimate extreme cases
- Model needs these examples for robust classification
- Recommend keeping outliers

### Scenario 3: No Significant Change
- Outliers have minimal impact on this particular model
- Either approach (before/after removal) is acceptable
- Decision can be based on other factors

---

## Advanced Customizations

You can modify the notebook for further analysis:

### Different Classification Algorithms
Replace Logistic Regression with:
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Gradient Boosting
- Neural Networks

### Different Outlier Detection Methods
- Z-score method
- Isolation Forest
- Local Outlier Factor (LOF)
- Robust PCA

### Hyperparameter Tuning
- GridSearchCV for finding optimal hyperparameters
- Cross-validation for better model evaluation

### Feature Engineering
- Create new features from existing ones
- Feature selection based on importance scores
- Dimensionality reduction (PCA)

---

## Requirements Breakdown

| Library | Purpose | Version |
|---------|---------|---------|
| pandas | Data manipulation & analysis | Latest |
| numpy | Numerical computing | Latest |
| matplotlib | Plotting & visualization | Latest |
| seaborn | Statistical data visualization | Latest |
| scikit-learn | Machine learning algorithms | Latest |
| jupyter | Interactive notebook environment | Latest |

---

## Tips for Running the Notebook

1. **Run cells sequentially:** Execute cells from top to bottom for proper variable initialization
2. **Monitor output:** Check console output for any warnings or errors
3. **Visualizations:** All plots will display inline in the notebook
4. **Execution time:** The full notebook takes 2-5 minutes to run
5. **Memory usage:** Ensure at least 1GB RAM available

---

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Run `pip install -r requirements.txt` to install all dependencies

### Issue: FileNotFoundError for spambase.data
**Solution:** Ensure the file path in the notebook matches your directory structure. Update line in Step 2:
```python
df = pd.read_csv("spambase Dataset/spambase.data", header=None)
```

### Issue: Kernel dies or slow execution
**Solution:** Close other applications to free up memory, or restart the kernel

---

## Project Summary

This comprehensive project demonstrates:
- Complete data preprocessing workflow
- Exploratory Data Analysis techniques
- Supervised learning classification
- Statistical outlier detection
- Model evaluation and comparison
- Data visualization best practices

The main takeaway is understanding how **data quality impacts model performance** and the importance of careful outlier analysis before drawing conclusions.

---

## References

- **Spambase Dataset:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/spambase)
- **Scikit-learn Documentation:** [sklearn.org](https://scikit-learn.org/)
- **Pandas Documentation:** [pandas.pydata.org](https://pandas.pydata.org/)

---

## Author Notes

This project is designed for:
- ✅ Learning complete ML workflows
- ✅ Understanding data preprocessing importance
- ✅ Comparing model performance
- ✅ Visualizing classification results
- ✅ Practicing with real-world datasets

---

**Created:** 2026
**Dataset:** Spambase (4,601 samples, 57 features)
**Algorithm:** Logistic Regression
**Focus:** Impact of Outlier Removal on Classification Performance
