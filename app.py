
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
@cross_origin()
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
@cross_origin()
def predict_api():
    data = request.json['data']

    new_data = list(data.values())
    output = model.predict([new_data])
    #print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = [float(i) for i in request.form.values()]
    final_features = [np.array(data)]
    output = model.predict(final_features)[0]
    print(output)
    return render_template('home.html', prediction_text="Airfoil pressure is  {}".format(output))


if __name__ == '__main__':
    app.run(debug=True)