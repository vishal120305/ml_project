import pickle
import os
import sys
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)
@app.route('/python_version', methods=['GET'])
def python_version():
    return sys.version

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            writing_score=request.form.get('writing_score'),
            reading_score=request.form.get('reading_score')
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df) 

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    
    predictions = []
    if request.method == 'POST':
        data = request.json
        # Preprocess the data
        # Make predictions using the trained model
        # Return the predictions as a JSON response
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)