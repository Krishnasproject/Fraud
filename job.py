from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy

import pandas as pd
import numpy as np
import logging
import os
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
from flask_mail import Mail, Message
mail = Mail()

import requests

DATA_PATH = 'adaptive_fraud_dataset.csv'
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production-' + str(hash(__file__)))

# Database configuration
database_url = os.getenv('DATABASE_URL', '')
if database_url:
    database_url = database_url.replace('postgres://', 'postgresql://')
app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///fraud_detection.db'
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() in ('true', '1', 't')
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')

mail.init_app(app)


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'register'
# Use in-memory storage for Flask-Limiter (works without Redis)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 5,
    'max_overflow': 2,
    'pool_timeout': 30,
    'pool_recycle': 1800
}
db = SQLAlchemy(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    whatsapp_number = db.Column(db.String(15), nullable=True)
    email = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

with app.app_context():
    db.create_all()

# Try to load pre-trained models
models_loaded = False
try:
    rf_model = joblib.load('rf_model.pkl')
    anomaly_model = joblib.load('anomaly_model.pkl')
    scaler = joblib.load('scaler.pkl')
    models_loaded = True
    logging.info("Loaded pre-trained models.")
except Exception as e:
    logging.error(f"Error loading models: {str(e)}")
    logging.info("Will train new models from dataset.")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    anomaly_model = IsolationForest(contamination=0.05, random_state=42)
    scaler = StandardScaler()

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    logging.info("Loaded existing dataset.")
else:
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({
        'TransactionID': range(1, n + 1),
        'AccountID': np.random.randint(1000, 5000, size=n),
        'TransactionAmount': np.random.uniform(10, 5000, size=n),
        'TransactionType': np.random.choice(['Transfer', 'Withdrawal', 'Deposit'], size=n),
        'AccountBalance': np.random.uniform(100, 10000, size=n),
        'HourOfTransaction': np.random.randint(0, 24, size=n),
        'Longitude': np.random.uniform(-180, 180, size=n),
        'Latitude': np.random.uniform(-90, 90, size=n),
        'FraudulentTransaction': np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    })
    df.to_csv(DATA_PATH, index=False)
    logging.info("Generated new dataset.")

account_avg_spend = df.groupby('AccountID')['TransactionAmount'].median().to_dict()
df['AvgTransactionAmount'] = df['AccountID'].map(account_avg_spend)
df['DeviationFromAvg'] = abs(df['TransactionAmount'] - df['AvgTransactionAmount'])
df['Overdraft'] = (df['TransactionAmount'] > df['AccountBalance']).astype(int)
df['TransactionType'] = df['TransactionType'].map({'Transfer': 0, 'Withdrawal': 1, 'Deposit': 2})

# Fit scaler if models weren't loaded
if not models_loaded:
    scaler.fit(df[['TransactionAmount', 'AccountBalance', 'AvgTransactionAmount', 'DeviationFromAvg']])
    joblib.dump(scaler, 'scaler.pkl')

df[['TransactionAmount', 'AccountBalance', 'AvgTransactionAmount', 'DeviationFromAvg']] = scaler.transform(
    df[['TransactionAmount', 'AccountBalance', 'AvgTransactionAmount', 'DeviationFromAvg']]
)

X = df.drop(columns=['FraudulentTransaction', 'TransactionID'])
y = df['FraudulentTransaction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

results = {}
rf_model.fit(X_train, y_train)
accuracy = rf_model.score(X_test, y_test)
results['Random Forest'] = accuracy

anomaly_model.fit(X_train)
predictions = anomaly_model.predict(X_test)
accuracy = (predictions == y_test).mean()
results['Isolation Forest'] = accuracy

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        whatsapp_number = request.form['whatsapp']
        email = request.form['email']
        if db.session.query(User).filter_by(username=username).first():
            return "Username already exists!"
        if not whatsapp_number or not email:
            return "WhatsApp number and email are required!"
        user = User(username=username, password_hash=generate_password_hash(password), whatsapp_number=whatsapp_number, email=email)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    logging.info(f"Login request method: {request.method}")
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        logging.info(f"Login attempt for username: {username}")
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('predict'))
        return "Invalid credentials!"
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/')

def index():
    return redirect(url_for('register'))

def send_whatsapp_notification(data, reasons):
    whatsapp_number = data.get('whatsapp_number')
    message = f"Fraud Alert!\nTransaction Amount: {data['TransactionAmount']}\nAccount ID: {data['AccountID']}\nReasons: {', '.join(reasons)}"
    api_url = f"https://api.callmebot.com/whatsapp.php?phone={whatsapp_number}&text={message}&apikey=YOUR_API_KEY"
    response = requests.get(api_url)
    if response.status_code == 200:
        logging.info("WhatsApp notification sent successfully.")
    else:
        logging.error(f"Failed to send WhatsApp notification: {response.text}")

@app.route('/predict', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
@login_required
def predict():
    try:
        if request.method == 'GET':
            return render_template('predict.html')

        if request.content_type != 'application/json':
            return jsonify({'error': 'Unsupported Media Type'}), 415

        data = request.get_json()
        logging.info(f"Incoming data from {request.remote_addr}: {data}")

        if data is None:
            return jsonify({'error': 'No data provided'}), 400

        data['TransactionAmount'] = float(data['TransactionAmount'])
        data['AccountBalance'] = float(data['AccountBalance'])
        data['Longitude'] = float(data['Longitude'])
        data['Latitude'] = float(data['Latitude'])

        account_id = data['AccountID']
        if account_id in account_avg_spend:
            data['AvgTransactionAmount'] = account_avg_spend[account_id]
        else:
            data['AvgTransactionAmount'] = df['TransactionAmount'].median()
            logging.info(f"Account ID {account_id} not found in training data, using global median transaction amount.")

        data['DeviationFromAvg'] = abs(data['TransactionAmount'] - data['AvgTransactionAmount'])
        data['Overdraft'] = int(data['TransactionAmount'] > data['AccountBalance'])

        reasons = []
        if data['TransactionAmount'] > data['AccountBalance'] * 10:
            reasons.append("Extremely high transaction amount compared to account balance")
        if data['Overdraft'] == 1:
            reasons.append("Transaction amount exceeds available balance")
        if data['DeviationFromAvg'] > df['DeviationFromAvg'].quantile(0.95):
            reasons.append("Transaction amount significantly deviates from account's typical amount")
        if data['HourOfTransaction'] in [0, 1, 2, 3, 4]:
            reasons.append("Transaction occurred during high-risk hours")



        if 'TransactionType' not in data:
            return jsonify({'error': 'TransactionType is required'}), 400
        
        transaction_type = data['TransactionType']
        if isinstance(transaction_type, str):
            mapped_type = {'Transfer': 0, 'Withdrawal': 1, 'Deposit': 2}.get(transaction_type)
            if mapped_type is None:
                logging.error(f"Invalid TransactionType received: {transaction_type}")
                return jsonify({'error': 'Invalid TransactionType'}), 400
            data['TransactionType'] = mapped_type
        else:
            return jsonify({'error': 'TransactionType must be a string'}), 400

        input_data = pd.DataFrame([data])
        input_data[['TransactionAmount', 'AccountBalance', 'AvgTransactionAmount', 'DeviationFromAvg']] = scaler.transform(
            input_data[['TransactionAmount', 'AccountBalance', 'AvgTransactionAmount', 'DeviationFromAvg']]
        )

        input_data = input_data[X.columns]

        prediction = rf_model.predict(input_data)
        anomaly_score = (anomaly_model.decision_function(input_data) + 1) / 2
        is_anomalous = anomaly_model.predict(input_data) == -1
        # Only flag as fraud if both models agree or there are clear fraud indicators
        is_fraud = bool(prediction[0]) and is_anomalous[0]
        
        # If models disagree but there are strong fraud indicators
        if (bool(prediction[0]) or is_anomalous[0]) and len(reasons) >= 2:
            is_fraud = True

        if is_anomalous[0]:
            reasons.append('Transaction detected as an anomaly by the Isolation Forest model.')

        logging.info(f"Prediction inputs: {input_data.to_dict()}")

        logging.info(f"RF prediction: {prediction[0]}, Anomaly score: {anomaly_score[0]}")
        logging.info(f"Final fraud decision: {is_fraud}, Reasons: {reasons}")

        if is_fraud and not reasons:
            reasons.append("Transaction flagged by fraud detection models")
        elif not is_fraud:
            reasons.append("Transaction appears valid based on all checks")




        # Send email notification for both fraud and non-fraud cases
        try:
            subject = "Fraud Detection Alert" if is_fraud else "Transaction Processed"
            status = "flagged as fraudulent" if is_fraud else "processed successfully"
            msg = Message(
                subject,
                recipients=[current_user.email],
                body=f"""Transaction Status: {status}
Transaction Amount: {data['TransactionAmount']}
Account ID: {data['AccountID']}
Reasons: {', '.join(reasons) if reasons else 'No issues detected'}"""
            )
            mail.send(msg)
            logging.info(f"Email notification sent to {current_user.email}")
            logging.debug(f"Mail server config: {app.config['MAIL_SERVER']}:{app.config['MAIL_PORT']}")
            logging.debug(f"Using TLS: {app.config['MAIL_USE_TLS']}")
            logging.debug(f"Auth username: {app.config['MAIL_USERNAME']}")

        except Exception as e:
            logging.error(f"Failed to send email notification: {str(e)}", exc_info=True)

        if is_fraud:
            send_whatsapp_notification(data, reasons)



        return jsonify({
            'Fraudulent': 'Yes' if is_fraud else 'No',
            'AnomalyScore': round(anomaly_score[0], 4),
            'Reasons': reasons
        })
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    app.run()
