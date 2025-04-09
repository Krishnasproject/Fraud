# Fraud Detection System

## Overview
This project implements a fraud detection system using machine learning techniques. It utilizes a Flask web application to serve predictions based on transaction data.

## Requirements
- Python 3.x
- Flask
- pandas
- numpy
- scikit-learn
- flask-limiter

## Setup
1. Clone the repository.
2. Navigate to the project directory.
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application
To run the application, execute:
```
python job.py
```
Then, open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage
Fill in the required fields and click "Predict Fraud" to check if a transaction is fraudulent. The application will provide reasons for the prediction.
