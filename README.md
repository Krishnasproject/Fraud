Fraud Detection System using Machine Learning

📌 Overview

This project is a Machine Learning–based Fraud Detection System built to identify potentially fraudulent financial transactions.
It uses multiple ML algorithms, compares their performance, and serves predictions through a simple Flask web application.

The goal of this project is to demonstrate:

End-to-end ML workflow

Model comparison and evaluation

Deployment-ready backend structure

🚀 Features

Trained multiple ML models for fraud detection

Compared models using accuracy and performance metrics

REST-based prediction system using Flask

Simple and clean UI for testing transactions

Modular and scalable project structure

🧠 Machine Learning Models Used

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

Naive Bayes

The best-performing model was selected based on evaluation metrics such as accuracy and consistency.

📊 Dataset

The dataset contains anonymized transaction data with relevant numerical features used for fraud classification.

⚠️ Note:
Dataset and trained model files are not included in the repository to keep the repo lightweight and to avoid sharing large or sensitive files.

🛠 Tech Stack

Python

Flask

scikit-learn

Pandas & NumPy

HTML / CSS

SQLite (optional for logging)

⚙️ Project Structure
Fraud/
│
├── job.py                  # Main Flask application
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── templates/              # HTML files
├── static/                 # CSS / assets
├── models/                 # Trained ML models (excluded in repo)
└── data/                   # Dataset (excluded)

▶️ How to Run the Project
# Clone the repository
git clone https://github.com/Krishnasproject/Fraud.git

# Navigate to project folder
cd Fraud

# Install dependencies
pip install -r requirements.txt

# Run the application
python job.py


Then open:

http://127.0.0.1:5000/

📈 Future Improvements

Add real-time transaction streaming

Improve feature engineering

Deploy using Docker / Cloud (AWS, Render, Railway)

Add explainability using SHAP or LIME

🙌 Author

Krishnanand Jha
Aspiring Data Scientist | Machine Learning Enthusiast

