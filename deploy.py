from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Model load karna (jaise class me pickle.load kiya tha)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

# 1. Single sample prediction route
@app.route('/predict_sample', methods=['POST'])
def predict_sample():
    # 200 random features sample test ke liye
    sample_features = np.random.normal(loc=0.0, scale=3.0, size=(1, 200))
    prediction = model.predict(sample_features)
    prob = model.predict_proba(sample_features)[0][1]

    if prediction[0] == 1:
        res = "Customer WILL make a transaction (Probability: {:.2f}%)".format(prob * 100)
    else:
        res = "Customer will NOT make a transaction (Probability: {:.2f}%)".format(prob * 100)

    return render_template('index.html', pred=res)

# 2. Batch CSV prediction route
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return render_template('index.html', pred_csv="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', pred_csv="Please select a CSV file.")

    df_test = pd.read_csv(file)

    X_input = df_test.copy()
    if 'ID_code' in X_input.columns:
        X_input.drop(columns=['ID_code'], inplace=True)
    if 'target' in X_input.columns:
        X_input.drop(columns=['target'], inplace=True)

    preds = model.predict(X_input)
    res_msg = f"Batch Prediction Done for {len(preds)} customers! | Will Transact: {(preds == 1).sum()} | Will Not Transact: {(preds == 0).sum()}"

    return render_template('index.html', pred_csv=res_msg)

if __name__ == "__main__":
    app.run(debug=True)
