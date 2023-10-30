from flask import Flask, render_template, request
import pickle  # vamos usar para carregar o modelo
import numpy as np

app = Flask(__name__)


with open('cls_iris.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    dict_convert = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}

    sepal_len = request.form.get('sepal_len')
    sepal_wid = request.form.get('sepal_wid')
    petal_len = request.form.get('petal_len')
    petal_wid = request.form.get('petal_wid')
    
    datas = np.array([petal_len, petal_wid, sepal_len, sepal_wid]).astype('float')  # transformando os dados em uma lista numpy
    pred = model.predict(datas.reshape(1, -1))
    
    return render_template('predict.html', pred=dict_convert[pred[0]])


if __name__ == '__main__':
    app.run(debug=True)

