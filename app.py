# -*- coding: utf-8 -*-
"""
Created on Sat May 14 21:53:08 2022

@author: 91863
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def campus_placement_prediction():
    
    """Let's predict campus placement  
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: gender
        in: query
        type: number
        required: true
      - name: ssc_p
        in: query
        type: number
        required: true
      - name: hsc_p
        in: query
        type: number
        required: true
      - name: degree_p
        in: query
        type: number
        required: true
      - name: workex
        in: query
        type: number
        required: true
      - name: etest_p
        in: query
        type: number
        required: true
      - name: specialisation
        in: query
        type: number
        required: true
      - name: mba_p
        in: query
        type: number
        required: true
      - name: Arts
        in: query
        type: number
        required: true
      - name: Commerce
        in: query
        type: number
        required: true
      - name: Science
        in: query
        type: number
        required: true
      - name: Comm&Mgmt
        in: query
        type: number
        required: true
      - name: Others
        in: query
        type: number
        required: true
      - name: Sci&Tech
        in: query
        type: number
        required: true
                        
        
    responses:
        200:
            description: The output values
        
    """
    gender=request.args.get("gender")
    ssc_p=request.args.get("ssc_p")
    hsc_p=request.args.get("hsc_p")
    degree_p=request.args.get("degree_p")
    workex=request.args.get("workex")
    etest_p=request.args.get("etest_p")
    specialisation=request.args.get("specialisation")
    mba_p=request.args.get("mba_p")
    Arts=request.args.get("Arts")
    Commerce=request.args.get("Commerce")
    Science=request.args.get("Science")
    Others=request.args.get("Others")
    CommMgmt=request.args.get("Comm&Mgmt")
    SciTech=request.args.get("Sci&Tech")
    
    prediction=classifier.predict([[gender,ssc_p,hsc_p,degree_p,workex,etest_p,specialisation,mba_p,Arts,Commerce,Science,CommMgmt,Others,SciTech]])
    print(prediction)
    return "Hello The answer is"+str(prediction)


    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    
    return str(list(prediction))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080)
