# Dry Bean Classification Project

## Project Overview

This project focuses on developing a multi-class classification model to accurately classify seven different types of dry beans based on features such as shape, form, and structure. The dataset contains images of 13,611 beans, from which 16 features (12 dimensions and 4 shape-related attributes) were extracted. Machine learning and deep learning methods were employed to optimize classification performance.

## Dataset Information

The dataset used includes images of seven types of dry beans:
- Seker
- Barbunya
- Bombay
- Cali
- Dermosan
- Horoz
- Sira

### Features
- 12 dimensions (e.g., area, perimeter)
- 4 shape forms (e.g., eccentricity, convexity)

## Project Goals

1. **Multi-Class Classification**: Develop a classification model to categorize beans into one of the seven types.
2. **Ensemble Methods**: Implement models like Random Forest and Gradient Boosting to improve model accuracy.
3. **Deep Neural Networks**: Experiment with multi-layer perceptron (MLP) models and transfer learning techniques.
4. **Model Evaluation**: Assess accuracy, minimize overfitting, and apply cross-validation techniques.

## Methodology

1. **Data Preprocessing**: Cleaned and preprocessed the dataset for training, including scaling and handling missing values.
2. **Model Selection**:
   - **Random Forest** and **Gradient Boosting** models for ensemble learning.
   - **Multi-Layer Perceptron (MLP)** for deep learning approaches with logistic regression for classification.
3. **Hyperparameter Tuning**: Optimized parameters like number of trees, learning rates, and layers for best performance.
4. **Cross-Validation**: Applied 10-fold cross-validation to ensure robust model performance.
5. **Overfitting Prevention**: Used techniques such as early stopping and validation metrics to mitigate overfitting.

## Results

- **Random Forest** and **Gradient Boosting** performed well, especially in handling the multi-class classification task.
- **Deep Learning** models using multi-layer perceptrons, combined with logistic regression, achieved further improvements in accuracy.
- **Cross-validation** and **early stopping** were effective in preventing overfitting, resulting in a model that generalizes well to unseen data.

## Technologies Used

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **XGBoost**
- **TensorFlow / Keras** (for deep learning)
- **Matplotlib / Seaborn** (for visualizations)

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>

