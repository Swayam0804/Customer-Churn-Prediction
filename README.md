# Customer-Churn-Prediction

## Project Overview ##
- Customer churn refers to customers who stop using a company’s service.
- In this project, I built a machine learning model to predict whether a customer is likely to churn based on their service usage and account information.
- This type of problem is important for businesses because retaining existing customers is cheaper than acquiring new ones.

## Objective ##
- Predict whether a customer will churn or not
- Understand which factors contribute most to customer churn
- Build a classification model using machine learning

## Dataset ##
- Dataset used: **Telecom Customer Churn Dataset**
- Each row represents a customer
- Target variable: **Churn** (Yes / No)
  
## Model Used ##
**Random Forest Classifier**
  Chosen because it:
- Handles complex data well
- Reduces overfitting
- Provides feature importance

## Evaluation Metrics ##
The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score

Recall was prioritized to correctly identify churn customers.

## Key Features Influencing Churn ##
- Contract type
- Tenure
- Monthly charges
- Total charges

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
