from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import kagglehub

app = Flask(__name__)

def download_and_prepare_data():
    """Download the dataset if not exists and prepare it for training"""
    if not os.path.exists('diabetes_prediction_dataset.csv'):
        path = kagglehub.dataset_download("iammustafatz/diabetes-prediction-dataset")
        csv_file = os.path.join(path, "diabetes_prediction_dataset.csv")
        df = pd.read_csv(csv_file)
        df.to_csv('diabetes_prediction_dataset.csv', index=False)
    else:
        df = pd.read_csv('diabetes_prediction_dataset.csv')
    return df

def load_and_preprocess_data():
    """Load and preprocess the data"""
    df = download_and_prepare_data()
    
    # Handle categorical variables
    le_gender = LabelEncoder()
    le_smoking = LabelEncoder()
    
    # Replace 'No Info' with 'Unknown' in smoking_history
    df['smoking_history'] = df['smoking_history'].replace('No Info', 'Unknown')
    
    df['gender'] = le_gender.fit_transform(df['gender'])
    df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])
    
    # Save encoders for prediction
    joblib.dump(le_gender, 'le_gender.joblib')
    joblib.dump(le_smoking, 'le_smoking.joblib')
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Save scaler for prediction
    joblib.dump(scaler, 'scaler.joblib')
    
    return df

def train_model(df):
    """Train the model and save it"""
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'model.joblib')
    
    # Calculate and print accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([data])
        
        # Load preprocessors
        le_gender = joblib.load('le_gender.joblib')
        le_smoking = joblib.load('le_smoking.joblib')
        scaler = joblib.load('scaler.joblib')
        model = joblib.load('model.joblib')
        
        # Preprocess input data
        input_df['gender'] = le_gender.transform(input_df['gender'])
        input_df['smoking_history'] = le_smoking.transform(input_df['smoking_history'])
        
        numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/retrain', methods=['POST'])
def retrain():
    """Endpoint to retrain the model with fresh data"""
    try:
        df = load_and_preprocess_data()
        model = train_model(df)
        return jsonify({'message': 'Model retrained successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Load preprocessed data and train model on startup if model doesn't exist
    if not os.path.exists('model.joblib'):
        df = load_and_preprocess_data()
        model = train_model(df)
    app.run(debug=True, host='0.0.0.0', port=5001)
