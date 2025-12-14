# Customer-Churn-Prediction
A supervised Machine Learning project focused on analyzing customer behavior and accurately predicting customer churn for a telecommunications company. This project employs a Random Forest Classifier to identify high-risk customers, allowing the business to implement proactive retention strategies.

## Features  
- Implemented a full machine learning pipeline from raw data handling to final model deployment.
- Utilized the class_weight='balanced' parameter within the Random Forest model to prevent bias and improve the model's ability to identify the minority (churning) class.
- Used Feature Importance analysis to identify the top behavioral drivers of churn (e.g., contract type, tenure, monthly charges).
- Optimized for high Recall and F1-Score to minimize False Negatives (missing a customer who is about to churn), which is critical for business revenue protection.
- Performed custom cleaning on the TotalCharges column and utilized LabelEncoder for categorical feature transformation.

## Requirements  
- Install dependencies:
   ```sh
      pip install numpy matplotlib scikit-learn

## How to Run
1. Clone the repository to your local machine :
   ```sh
   git clone https://github.com/Swayam0804/Customer-Churn-Prediction.git

2. Navigate to the project directory :
   ```sh
   cd Customer-Churn-Prediction

3. Run the Jupyter Notebook:
   View the complete analysis directly on Kaggle: [https://www.kaggle.com/code/swyamsreepatra/customer-churn-prediction]
   ```sh
   jupyter notebook customer-churn-prediction-ipynb.ipynb
