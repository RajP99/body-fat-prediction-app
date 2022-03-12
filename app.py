from flask import Flask, jsonify, request
import pickle
import os
import traceback
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return "hey"

@app.route('/predict', methods=['POST'])
def predict():
	try:
		json = request.get_json()	 
		temp=list(json[0].values())
		temp = np.array([temp])
		SC=StandardScaler()
		temp = SC.transform(temp)
		rfr = pickle.load(open('rfr.pkl', 'rb'))
		prediction = rfr.predict(temp)
		print("Prediction: ", prediction)        
		return jsonify({'prediction': str(prediction)})

	except:        
		return jsonify({'trace': traceback.format_exc()})
    


if __name__ == '__main__':
    app.run(debug=True)
