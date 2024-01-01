from flask import Flask , render_template , request
import pandas as pd
import pickle
import numpy as np
pipe = pickle.load(open('LRidgeModel.pkl','rb'))
n = pd.read_csv('n.csv')
app = Flask(__name__)
@app.route('/')
def index():
    location = sorted(n['location'].unique())
    return render_template('index.html',location=location)

@app.route('/p',methods=['POST'])
def p():
    location  = request.form.get('location')
    bhk = request.form.get('bhk')
    bedroom = request.form.get('bedroom')
    swg = request.form.get('swg')
    input = pd.DataFrame([[location, swg, bedroom, bhk]], columns=['location', 'total_sqft', 'bath','bhk'])
    prediction = pipe.predict(input)[0] * 100000 

    return str(np.round(prediction,2))




if __name__  + "__main__":
    app.run(debug=True,port=5001)