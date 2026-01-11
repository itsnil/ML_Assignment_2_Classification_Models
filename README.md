# Early Stage Diabetes Risk Prediction

## Problem Statement
The objective is to predict whether a patient is at risk of diabetes based on diagnostic measures and symptoms. This project compares 6 different Machine Learning algorithms to find the most accurate model.

## Dataset Description
- **Dataset**: UCI Diabetes Data Upload
- **Features**: 16 (Age, Gender, Polyuria, Polydipsia, etc.)
- **Instances**: 520
- **Target**: Class (Positive/Negative)

## Model Train-Test Split Information
All models were trained on 80% of the data and tested on 20%. 

## Models Compared
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

## Model Evaluation Metrics
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC Score)

## Models vs Evaluation metrics

| **ML Model Name** | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** 
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.923077 | 0.971404 |  0.931507 | 0.957746 | 0.944444 | 0.820358 |
| **Decision Tree** | 0.951923 | 0.964789 | 1.000000 | 0.929577 | 0.963504 | 0.898479 |
| **KNN** | 0.855769 | 0.948570 | 0.951613 | 0.830986 | 0.887218 | 0.702009 |
| **Naive Bayes** | 0.913462 | 0.960734 | 0.930556 | 0.943662 | 0.937063 | 0.798823 |
| **Random Forest** | 0.990385 | 1.000000 | 1.000000 | 0.985915 | 0.992908 | 0.978222 |
| **XGBoost** | 0.980769 | 0.999146 | 1.000000 | 0.971831 | 0.985714 | 0.957234 | 


## Model Performance Observations

| ML Model Name | Observations About Model Performance |
| :--- | :--- |
| **Logistic Regression** | This model provided a solid baseline with high Recall (~96%), indicating it is quite sensitive to detecting positive diabetes cases. Its AUC of 0.97 shows excellent capability in distinguishing between classes, though it was slightly outperformed by tree-based methods. |
| **Decision Tree** | Achieved a perfect Precision score of 1.0, meaning every patient it predicted as "Positive" actually had diabetes (zero False Positives). However, its Recall (93%) was lower than the ensemble methods, meaning it missed a few actual positive cases. |
| **KNN** | The weakest performer on this dataset with the lowest Accuracy (85.6%) and Recall (83.1%). This suggests that simple distance-based classification was less effective here compared to decision boundaries created by tree-based models. |
| **Naive Bayes** | Performed very similarly to Logistic Regression with balanced Precision and Recall (~93-94%). While reliable and fast, it did not capture the complex patterns as effectively as the ensemble models. |
| **Random Forest (Ensemble)** | **The Best Performing Model.** It achieved the highest scores across almost all metrics, including 99.04% Accuracy and perfect Precision. Its high Recall (98.6%) makes it the safest choice for medical diagnosis, as it misses the fewest positive cases. |
| **XGBoost (Ensemble)** | An extremely strong runner-up with 98.08% Accuracy. Like the Random Forest and Decision Tree, it achieved perfect Precision (1.0). It is a highly effective model for this data, only slightly trailing Random Forest in Recall sensitivity. |
