from flask import Flask,render_template,request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    input_data = pd.DataFrame(columns=columns)

    input_data.loc[0] = 0


    input_data["Age"] = float(request.form['Age'])
    input_data["MonthlyIncome"] = float(request.form['MonthlyIncome'])
    input_data["JobSatisfaction"] = float(request.form['JobSatisfaction'])
    input_data["YearsAtCompany"] = float(request.form['YearsAtCompany'])


    # One-hot encoded columns

    jobrole = request.form['JobRole']
    overtime = request.form['OverTime']
    travel = request.form['BusinessTravel']


    jobrole_col = "JobRole_" + jobrole
    overtime_col = "OverTime_" + overtime
    travel_col = "BusinessTravel_" + travel


    if jobrole_col in input_data.columns:
        input_data[jobrole_col] = 1

    if overtime_col in input_data.columns:
        input_data[overtime_col] = 1

    if travel_col in input_data.columns:
        input_data[travel_col] = 1


    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)


    if prediction[0]==1:
        result="Employee Will Leave"
    else:
        result="Employee Will Stay"


    return render_template('index.html',prediction=result)



if __name__=="__main__":
    app.run(debug=True)