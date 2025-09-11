import pickle
import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)
CORS(app)

@app.route('/python_version', methods=['GET'])
def python_version():
    return sys.version

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return jsonify({'message': 'Send a POST request with student data'})
    else:
        try:
            # Get JSON data from request
            data = request.get_json()
            
            # Create CustomData object
            student_data = CustomData(
                gender=data.get('gender'),
                race_ethnicity=data.get('race_ethnicity'),
                parental_level_of_education=data.get('parental_level_of_education'),
                lunch=data.get('lunch'),
                test_preparation_course=data.get('test_preparation_course'),
                writing_score=float(data.get('writing_score')),
                reading_score=float(data.get('reading_score'))
            )
            
            # Get data as DataFrame and predict
            pred_df = student_data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            # Return JSON response
            return jsonify({
                'prediction': float(results[0]),
                'status': 'success'
            })
            
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 400
        
    

if __name__ == '__main__':
    app.run(debug=True)