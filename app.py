from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from helper import Helper

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        uploaded_file = request.files['imageFile']
        filename = secure_filename(uploaded_file.filename)
        uploaded_file.save("static/uploaded_img/" + filename)

        prediction, prob = Helper().classification(filename)
        return render_template('result.html', prediction=prediction, prob=prob, filename=filename)

app.run(host="localhost", port=8080, debug=True)