#!/usr/bin/python3
from flask import Flask
from flask import send_file, make_response, send_from_directory,request
import os
import io
import json
import sys
import numpy 
import JudgePrediction
from sklearn.externals import joblib

properties_path = '\db_properties\db_connection.json' #, '2018-05-16')
boost = joblib.load('model.pkl')
app = Flask(__name__)

   

@app.route("/")
def root():
    return send_from_directory(".", "index.html")
    # return "Welcome!"
    # return app.send_static_file('index.html')

#@app.route("/image/")
#def images():
#	return json.dumps()
    

@app.route("/prediction/<int:caseId>/<int:judgeId>", methods=['GET'])
def prediction(caseId, judgeId):
    return "{}".format( JudgePrediction.predict(boost, caseId, judgeId, properties_path))




if __name__ == "__main__":
    app.run(debug=True, host= '0.0.0.0')
