from flask import Flask
from flask import request
from flask import render_template
import pickle

file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()
app = Flask(__name__)

@app.route('/',methods=["GET","POST"])
def Hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])

        input_features = [fever, pain,age, runnyNose, diffBreath]
        infProb = clf.predict_proba([input_features])[0][1]
        print(infProb)
        return render_template('show.html',inf=(infProb)*100)
    return render_template('index.html')


if __name__  == "__main__":
    app.run(debug=True)