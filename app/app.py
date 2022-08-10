from crypt import methods
from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import numpy as np

app = Flask(__name__)

def drop_down_to_list(options, selection):
    val = []
    for option in options:
        if(option == selection):
            val.append(1)
        else:
            val.append(0)
    
    return val

@app.route('/', methods=['GET', 'POST'])
def stroke_prediction():
    request_type = request.method
    if(request_type == "GET"):
        return render_template('index.html')
    else:
        user_input_form = request.form

        user_height = float(user_input_form['height'])/100
        user_weight = float(user_input_form['weight'])
        user_bmi = user_weight/pow(user_height, 2)

        user_input = []

        user_input.append(float(user_input_form['age']))
        user_input.append(float(user_input_form['hypertension']))
        user_input.append(float(user_input_form['heart_disease']))
        user_input.append(float(user_input_form['avg_glucose_level']))
        user_input.append(user_bmi)

        genders = ['Female', 'Male', 'Other']
        user_input += drop_down_to_list(genders, user_input_form['gender'])

        married = ['No', 'Yes']
        user_input += drop_down_to_list(married, user_input_form['ever_married'])

        work = ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children']
        user_input += drop_down_to_list(work, user_input_form['work_type'])

        residence = ['Rural', 'Urban']
        user_input += drop_down_to_list(residence, user_input_form['Residence_type'])

        smoking = ['formerly_smoked', 'never_smoked', 'smokes']
        user_input += [0]
        user_input += drop_down_to_list(smoking, user_input_form['smoking_status'])

        user_input = np.array(user_input)

        sc = load('app/scaler.joblib')
        user_input_scaled = sc.transform (user_input.reshape(1, -1))

        model = load('app/stroke_pred_model.joblib')

        prediction = model.predict(user_input_scaled)

        output = ""

        if(prediction[0] == 0):
            output = "You are not at the risk of having a stroke"
        elif(prediction[0] == 1):
            output = "You are at the risk of having a stroke"

        return render_template('index.html', result=output)

if __name__ == '__main__':
    app.run()