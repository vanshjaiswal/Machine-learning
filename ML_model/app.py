from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open("model.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_input=[int(x) for x in request.form.values()]
    array_input=[np.array(int_input)]
    prediction=model.predict(array_input)
    
    return render_template('index.html', prediction_text='person is {}'.format(prediction) )

# @app.route('/results', methods=['POST'])
#def results():
 #   data=request.get_json(force=True)
  #  prediction=model.predict([np.array(data.values())])

   # return jsonify(prediction) 
