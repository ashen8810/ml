import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)


@app.route('/')
def home():

    return render_template('home.html')

@app.route('/index',methods = ["GET","POST"])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == "POST":
        # features = [x for x in request.form.values()]
        # print(features)
        # print(request.form.keys())
        f = []
        features = [request.form['Nausea'],request.form["Photophobia"],request.form["Phonophobia"],request.form["Aggravation"],request.form['conjunctival_injection'],
             request.form['lacrimation'],request.form["nasal_congestion"],request.form['rhinorrhoea'],request.form['eyelid_oedema'],request.form['sweating'],request.form['miosis'],
             request.form['ptosis'],request.form['Speech disturbance'],request.form['Visual symptoms'],request.form['Sensory symptoms'],request.form['homonymous_symptomps'],
             request.form['dysarthria'],request.form["hemiplegic"]]
        for i in features:
            f.append(int(i))

        if request.form["Location"]=="1":
            f.extend([0,0,1])
        if request.form["Location"]=="2":
            f.extend([1, 0, 0])
        if request.form["Location"]=="3":
            f.extend([0,1,0])
        if request.form["Severity"]=="1":
            f.extend([0,0,1])
        if request.form["Severity"]=="2":
            f.extend([0,1,0])
        if request.form["Severity"]=="3":
            f.extend([1, 0, 0])
        if request.form["Characterisation"]=="1":
            f.extend([0,0,1])
        if request.form["Characterisation"]=="2":
            f.extend([1,0,0])
        if request.form["Characterisation"]=="3":
            f.extend([0,1,0])
        if request.form["Duration"]=="1":
            f.extend([1,0,0,0,0,0,0,0])
        if request.form["Duration"]=="2":
            f.extend([0,1,0,0,0,0,0,0])
        if request.form["Duration"]=="3":
            f.extend([0,0,1,0,0,0,0,0])
        if request.form["Duration"]=="4":
            f.extend([0,0,0,1,0,0,0,0])
        if request.form["Duration"]=="5":
            f.extend([0,0,0,0,1,0,0,0])
        if request.form["Duration"]=="6":
            f.extend([0,0,0,0,0,1,0,0])
        if request.form["Duration"]=="7":
            f.extend([0,0,0,0,0,0,1,0])
        if request.form["Duration"]=="8":
            f.extend([0,0,0,0,0,0,0,1])
        if request.form["Prev Attacks"]=="1":
            f.extend([0,1,0,0])
        if request.form["Prev Attacks"]=="2":
            f.extend([0,0,0,1])
        if request.form["Prev Attacks"]=="3":
            f.extend([1,0,0,0])
        if request.form["Prev Attacks"]=="4":
            f.extend([0,0,1,0])

        # print(len(f))
        # print(model)
        import pickle, joblib
        filename = "finalized_model1.sav"
        try:
            loaded_model = joblib.load(filename)
            output = loaded_model.predict([f])
            print(output)





            # f = request.form["Nausea"]
            print(f)

            return render_template('result.html', prediction_text='{}'.format(output[0]))
        except ValueError as e:
            print(e)
            return render_template('index.html', prediction_text="Input values are not valid")


    else:

        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)

