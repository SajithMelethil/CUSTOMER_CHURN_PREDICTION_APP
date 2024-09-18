from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder='template')
adaboost = pickle.load(open('adaboost.pkl', 'rb'))

@app.route('/')
def home() -> str:
    return render_template("homepage.html")

def get_data() -> pd.DataFrame:
    # Collect form data with error handling
    try:
        tenure = float(request.form.get('tenure', 0))
        MonthlyCharges = float(request.form.get('MonthlyCharges', 0))
        TotalCharges = float(request.form.get('TotalCharges', 0))
        gender = request.form.get('gender')
        SeniorCitizen = int(request.form.get('SeniorCitizen', 0))
        Partner = request.form.get('Partner')
        Dependents = request.form.get('Dependents')
        PhoneService = request.form.get('PhoneService')
        MultipleLines = request.form.get('MultipleLines')
        InternetService = request.form.get('InternetService')
        OnlineSecurity = request.form.get('OnlineSecurity')
        OnlineBackup = request.form.get('OnlineBackup')
        DeviceProtection = request.form.get('DeviceProtection')
        TechSupport = request.form.get('TechSupport')
        StreamingTV = request.form.get('StreamingTV')
        StreamingMovies = request.form.get('StreamingMovies')
        Contract = request.form.get('Contract')
        PaperlessBilling = request.form.get('PaperlessBilling')
        PaymentMethod = request.form.get('PaymentMethod')

        # Prepare data dictionary with default values
        d_dict = {
            'tenure': [tenure],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges],
            'gender_Female': [1 if gender == 'Female' else 0],
            'gender_Male': [1 if gender == 'Male' else 0],
            'SeniorCitizen_0': [1 if SeniorCitizen == 0 else 0],
            'SeniorCitizen_1': [1 if SeniorCitizen == 1 else 0],
            'Partner_No': [1 if Partner == 'No' else 0],
            'Partner_Yes': [1 if Partner == 'Yes' else 0],
            'Dependents_No': [1 if Dependents == 'No' else 0],
            'Dependents_Yes': [1 if Dependents == 'Yes' else 0],
            'PhoneService_No': [1 if PhoneService == 'No' else 0],
            'PhoneService_Yes': [1 if PhoneService == 'Yes' else 0],
            'MultipleLines_No': [1 if MultipleLines == 'No' else 0],
            'MultipleLines_No phone service': [1 if MultipleLines == 'No phone service' else 0],
            'MultipleLines_Yes': [1 if MultipleLines == 'Yes' else 0],
            'InternetService_DSL': [1 if InternetService == 'DSL' else 0],
            'InternetService_Fiber optic': [1 if InternetService == 'Fiber optic' else 0],
            'InternetService_No': [1 if InternetService == 'No' else 0],
            'OnlineSecurity_No': [1 if OnlineSecurity == 'No' else 0],
            'OnlineSecurity_No internet service': [1 if OnlineSecurity == 'No internet service' else 0],
            'OnlineSecurity_Yes': [1 if OnlineSecurity == 'Yes' else 0],
            'OnlineBackup_No': [1 if OnlineBackup == 'No' else 0],
            'OnlineBackup_No internet service': [1 if OnlineBackup == 'No internet service' else 0],
            'OnlineBackup_Yes': [1 if OnlineBackup == 'Yes' else 0],
            'DeviceProtection_No': [1 if DeviceProtection == 'No' else 0],
            'DeviceProtection_No internet service': [1 if DeviceProtection == 'No internet service' else 0],
            'DeviceProtection_Yes': [1 if DeviceProtection == 'Yes' else 0],
            'TechSupport_No': [1 if TechSupport == 'No' else 0],
            'TechSupport_No internet service': [1 if TechSupport == 'No internet service' else 0],
            'TechSupport_Yes': [1 if TechSupport == 'Yes' else 0],
            'StreamingTV_No': [1 if StreamingTV == 'No' else 0],
            'StreamingTV_No internet service': [1 if StreamingTV == 'No internet service' else 0],
            'StreamingTV_Yes': [1 if StreamingTV == 'Yes' else 0],
            'StreamingMovies_No': [1 if StreamingMovies == 'No' else 0],
            'StreamingMovies_No internet service': [1 if StreamingMovies == 'No internet service' else 0],
            'StreamingMovies_Yes': [1 if StreamingMovies == 'Yes' else 0],
            # Contract and Billing methods
            **{f"Contract_{contract}": [int(Contract == contract)] for contract in ['Month-to-month', "One year", "Two year"]},
            **{f"PaperlessBilling_{billing}": [int(PaperlessBilling == billing)] for billing in ['No', "Yes"]},
            **{f"PaymentMethod_{method}": [int(PaymentMethod == method)] for method in ['Bank transfer (automatic)', 
                                                                                     "Credit card (automatic)", 
                                                                                     "Electronic check", 
                                                                                     "Mailed check"]}
        }

        return pd.DataFrame.from_dict(d_dict, orient='columns')

    except Exception as e:
        logging.error(f"Error in getting data: {e}")
        return pd.DataFrame() # Return an empty DataFrame on error

def feature_imp(model, data: pd.DataFrame) -> pd.DataFrame:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_30_indices = indices[:30]
    return data.iloc[:, top_30_indices]

def min_max_scale(data: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled)

@app.route('/send', methods=['POST'])
def show_data() -> str:
    df = get_data()
    # Check for empty DataFrame before proceeding
    if df.empty:
        return render_template('results.html', tables=[], result="Error processing input data.")

    featured_data = feature_imp(adaboost, df)
    scaled_data = min_max_scale(featured_data)
    
    prediction = adaboost.predict(scaled_data)
    
    outcome = "Churner" if prediction[0] == 1 else "Non-Churner"

    return render_template(
        "results.html",
        tables=[df.to_html(classes='data', header=True)],
        result=outcome
    )

if __name__=="__main__":
    app.run(debug=True)
