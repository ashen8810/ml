from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
#def index():
#    return render_template('index.html')
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        Age=int(request.form['Age'])
        Experience=int(request.form['Experience'])
        Income=float(request.form['Income'])
        Family=int(request.form['Family'])
        CCAvg=float(request.form['CCAvg'])
        Education=int(request.form['Education'])
        Mortgage=int(request.form['Mortgage'])
        SecuritiesAccount=int(request.form['SecuritiesAccount'])
        CDAccount=int(request.form['CDAccount'])
        Online=int(request.form['Online'])
        CreditCard=int(request.form['CreditCard'])
       
        
        input_dict={
            'Age' : Age, 
            'Experience' : Experience, 
            'Income' : Income, 
            'Family' : Family, 
            'CCAvg' : CCAvg, 
            'Education' : Education,
            'Mortgage' : Mortgage, 
            'SecuritiesAccount' : SecuritiesAccount, 
            'CDAccount' : CDAccount, 
            'Online' : Online, 
            'CreditCard' : CreditCard
        }

        obs=pd.DataFrame(input_dict,index=[0])

        pred = model.predict(obs)[0] # return 0 or 1
        pred_prob = model.predict_proba(obs)[:,1][0] # return the prob of accepting personal loan

        #output = round(prediction[0], 2)
        #output = prediction.astype('int')
        return render_template('result.html', prediction=pred)


if __name__ == "__main__":
    app.run(debug=True)