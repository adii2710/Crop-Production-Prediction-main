from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# 1990	1485.0	121.00	16.37	Albania	Maize	36613
# 1	1990	1485.0	121.00	16.37	Albania	Potatoes	66667
# 2	1990	1485.0	121.00	16.37	Albania	Rice, paddy	23333
# 3	1990	1485.0	121.00	16.37	Albania	Sorghum	12500

# loading models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    predicted_value = 0
    try:
        if request.method == 'POST':
            Year = request.form['Year']
            average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
            pesticides_tonnes = request.form['pesticides_tonnes']
            avg_temp = request.form['avg_temp']
            Area = request.form['Area'].title()
            Item = request.form['Item'].title()

            features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])

            transformed_features = preprocessor.transform(features)
            predicted_value = dtr.predict(transformed_features).reshape(1, -1)
            print(predicted_value)
            return render_template('index.html', predicted_value=predicted_value)
    except Exception as err:
        print(err, "error")
        return render_template('index.html', predicted_value=err)


# python main
if __name__ == "__main__":
    app.run(debug=True)
