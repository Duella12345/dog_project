# Importing required libs
from flask import Flask, redirect, render_template, request
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}
from model import human_dog_identifier
 
# Instantiating flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route("/")
def main():
    return render_template("index.html")
 
 
# Prediction route
@app.route('/', methods=['POST', 'GET'])
def predict_image_file():
    try:
        if request.method == 'POST':
            
            # check if the post request has the file part
            if 'file' not in request.files:
                print('No file part')
                return redirect(request.url)
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                print('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                img = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                return render_template("result.html", predictions=human_dog_identifier(img))
        return render_template("result.html")
 
    except Exception as e: 
        print(e)
        error = "File cannot be processed."
        return render_template("result.html", err=error)
    
 
 
# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)