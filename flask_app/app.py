# credit: created using https://geekpython.in/flask-app-for-image-recognition as template
# Importing required libs
import cv2
from flask import Flask, render_template, request
from matplotlib import pyplot as plt
from model import human_dog_identifier
 
# Instantiating flask app
app = Flask(__name__)
 
 
# Home route
@app.route("/")
def main():
    return render_template("index.html")
 
 
# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            
            img_path = request.files['file'].stream
            human_dog_identifier(img_path)
            # pred = predict_result(img)
        
            # load color (BGR) image
            img = cv2.imread(img_path)

            # convert BGR image to RGB for plotting
            cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # display the image, along with bounding box
            plt.imshow(cv_rgb)
            plt.show()

            return render_template("result.html", predictions=str(human_dog_identifier(img_path)))
 
    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)
    
 
 
# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)