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
![image](https://github.com/user-attachments/assets/a53586d9-b37d-4e2f-9ced-47dde465a347)
![image](https://github.com/user-attachments/assets/0b7cf11b-98b2-415a-8ce6-b910e2d59b2a)
![image](https://github.com/user-attachments/assets/6ff64641-589a-40a4-8212-6a14797958a3)
![image](https://github.com/user-attachments/assets/d231a94e-10cd-4c53-b0b0-64924cbde95b)
- **Deep Learning** models using multi-layer perceptrons, combined with logistic regression, achieved further improvements in accuracy.
- **Cross-validation** and **early stopping** were effective in preventing overfitting, resulting in a model that generalizes well to unseen data.
![image](https://github.com/user-attachments/assets/57f7a253-2989-4d0c-9033-5e2a50525788)

*We can see that:
1. At lower max_depth values (1-8), both the training and validation accuracies improve together, indicating that increasing the complexity of the tree helps the model better capture the patterns in the data without overfitting. 
2. As the depth continues to increase (from 8 onwards), the training accuracy stays high (near 1.0), but the validation accuracy starts to decline slightly or plateaus. This is a classic sign of overfitting: the model is becoming too complex, fitting the noise in the training data rather than generalizing to unseen data.
3. No Further Improvement Beyond Depth of 8: The best balance between training and validation performance is around a max_depth of 8. Beyond this point, deeper trees (max_depth values of 16, 32, etc.) do not provide any improvement to validation accuracy, and in fact, validation accuracy starts to decrease slightly, while training accuracy remains very high.

![image](https://github.com/user-attachments/assets/e07145b7-2140-4c44-9fd9-bf3480507e34)

*We can see that even though the initialized model allows up to 200 estimators, the algorithm only fit 95 estimators (over 95 rounds of training)


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

