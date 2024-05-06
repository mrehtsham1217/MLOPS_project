from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.mlops_project.pipelines.test_pipeline import PredictionPipelines, CustomerData

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_data', methods=['POST', 'GET'])
def predict_data():
    if request.method == 'GET':
        return render_template('input_forms.html')
    else:
        data = CustomerData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            writing_score=int(request.form.get('writing_score')),
            reading_score=int(request.form.get('reading_score')),
        )
        pred_df = data.get_data_frame()
        print(pred_df)
        pred_pipeline = PredictionPipelines()
        results = pred_pipeline.prediction(pred_df)
        return render_template('input_forms.html', results=results[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
