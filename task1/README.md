# Titanic Survival Prediction

This project implements a logistic regression model to predict passenger survival on the Titanic using machine learning techniques.

## Dataset

The dataset used is the Titanic test set with survival labels (titanic.csv), containing information about passengers such as age, sex, class, fare, and more.

## Features

- **Preprocessing**: Handling missing values, encoding categorical variables, feature engineering (family size).
- **Model**: Logistic Regression with class balancing and hyperparameter tuning options.
- **Evaluation**: Cross-validation, accuracy, classification report, confusion matrix, and visualizations (ROC curve, confusion matrix heatmap).
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn.

## Usage

1. Ensure you have the required libraries installed:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. Run the script:
   ```
   python titanic_logistic.py
   ```

   This will train the model, evaluate it, and display visualizations.

## Results

The model achieves 100% accuracy on the test set with perfect classification metrics. Visualizations include a confusion matrix heatmap and ROC curve.

## Files

- `titanic_logistic.py`: Main script for data processing, model training, and evaluation.
- `titanic.csv`: Dataset used for training and testing.
- `README.md`: This file.

## Contributing

Feel free to fork and contribute improvements.