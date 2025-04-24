# 🚗 Accident Fatality Predictor – AI Project (COMP 247)

This project is a supervised machine learning application developed as part of the COMP 247 course project. 
It predicts whether a road accident is likely to result in a **fatality** or not, using real-world collision data from Toronto Police Services.

## 📌 Purpose

- To analyze and model accident-related data to classify incidents as fatal or non-fatal.
- To deploy the model as a **web API** using Flask and build a simple frontend for inference.

## 📊 Features

- Data cleaning, visualization, and correlation analysis
- Handling missing values and normalizing data
- Feature selection and pipeline setup
- Model training using:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - Neural Network
- Model tuning with Grid Search
- Evaluation with:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix & ROC curve
- Web API built with Flask
- HTML frontend for user input and prediction

## 🛠️ Technologies Used

- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Flask (for API deployment)
- HTML + Jinja Templates

# Run the Flask web application
python app.py

# Access the web app at:
📍 http://127.0.0.1:5000/

📁 Dataset Info

Source: Toronto Police Service Open Data Portal
Dataset: KSI (Killed or Seriously Injured) Dataset 
📍 https://data.torontopolice.on.ca/pages/ksi

!! Note: The dataset CSV file is not included in the GitHub repo due to its size.
Please download it directly from the source link above to use it locally for development or retraining.

