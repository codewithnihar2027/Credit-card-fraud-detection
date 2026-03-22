# Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Fraud detection is a critical problem in the financial domain due to the extremely imbalanced nature of the dataset, where fraudulent transactions represent a very small fraction of total transactions.

The dataset used in this project is the ULB Credit Card Fraud Detection dataset from Kaggle, containing 284,807 transactions with only 492 fraud cases. The features are anonymized using PCA transformation (V1–V28), along with Time and Amount fields.

The project follows a complete machine learning pipeline starting from data preprocessing to model evaluation. Initially, the dataset was checked for missing values and duplicate records, and duplicates were removed to ensure data quality. Exploratory Data Analysis (EDA) was performed to understand class imbalance, transaction amount distribution, and feature relationships.

Since the dataset is highly imbalanced, SMOTE (Synthetic Minority Oversampling Technique) was applied after train-test splitting to balance the training data and avoid data leakage. Feature scaling was performed on the Time and Amount columns using StandardScaler.

Three machine learning models were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest

Model performance was evaluated using Precision, Recall, F1-score, and ROC-AUC score, as accuracy is not a reliable metric for imbalanced datasets.

Results:

Logistic Regression achieved high recall (~0.87), meaning it detected most fraud cases, but had very low precision (~0.05), leading to many false positives. Decision Tree provided a better balance between precision (~0.40) and recall (~0.71), but its overall performance was moderate. Random Forest achieved the best overall performance with higher precision (~0.54), strong recall (~0.79), and the highest ROC-AUC (~0.98), making it the most effective model for this problem.

In fraud detection systems, recall is more important than accuracy because missing fraudulent transactions can lead to significant financial loss. However, a balance between precision and recall is necessary to reduce false alarms, which is best handled by the Random Forest model in this case.

Conclusion:

Random Forest is the most suitable model for this fraud detection task as it provides a strong balance between detecting fraud and minimizing false positives. Logistic Regression is useful when maximizing fraud detection is the only priority, while Decision Tree offers a moderate trade-off.

Future Improvements:

* Hyperparameter tuning using GridSearchCV
* Implementation of advanced models like XGBoost
* Real-time fraud detection system development
* Deployment using Streamlit or Flask for practical usage
