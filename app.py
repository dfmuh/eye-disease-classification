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

        # Check if the uploaded file is an image (jpg, jpeg, png)
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            uploaded_file.save("static/uploaded_img/" + filename)
            prediction, prob = Helper().classification(filename)
            return render_template('result.html', prediction=prediction, prob=prob, filename=filename)

        # Check if the uploaded file is a CSV file
        elif filename.endswith('.csv'):
            uploaded_file.save("static/uploaded_csv/" + filename)
            predictions = Helper().classification_list(filename)
            return render_template('result_list.html', predictions=predictions, filename=filename)

app.run(host="localhost", port=8080, debug=True)