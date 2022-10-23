import random
import time

import flask
import pandas

from ModelOutputs import ModelOutputs
from DataExtractor import DataExtractor
import os
import numpy as np
import shutil

# Initialize the flask app
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = "temp"
HOME_PATH = r"C:\Users\Aiden\Desktop\Disease Recognition"

# Initialize the model outputs
modelOutputs = ModelOutputs()
modelOutputs.createModel("Melanoma", "./Models/Melanoma2-7.h5")
modelOutputs.createModel("SkinCancer", "./Models/HAM03.3.h5")
modelOutputs.createModel("CovidXray", "./Models/CXRY-Finalist01.h5")

# Extract data
dataExtractor = DataExtractor()

# Initialize the flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    return flask.render_template("Home.html")


@app.route("/MelanomaImageDetector", methods=["GET", "POST"])
def melanomaImageDetector():
    # If image is not provided, return the rendered template
    if flask.request.method == "GET":
        return flask.render_template("MelanomaImageDetector.html", message="Upload a JPG or PNG image")
    # If image is provided, get the image
    image = flask.request.files["image"]
    if image.filename == '':
        return flask.render_template("MelanomaImageDetector.html", message="No image selected")
    if image and allowed_file(image.filename):
        subfolder = str(random.randint(0, 999999))
        os.makedirs(app.config['UPLOAD_FOLDER'] + "/" + subfolder + "/Predict")
        path = os.path.join(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], subfolder), "Predict"), image.filename)
        image.save(path)

        # Check if the image has been uploaded into the temp folder
        for x in range(300):
            if os.path.isfile(path):
                break
            time.sleep(0.01)

        # Get the image prediction
        prediction = modelOutputs.getOutputImage("Melanoma", os.path.join(app.config['UPLOAD_FOLDER'], subfolder) + "\\")
        # Remove the subfolder
        shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], subfolder))
        # Return the rendered template
        return flask.render_template("MelanomaImageDetector.html", message="If you see this, something went wrong :/", pred=True, predictions=(round(prediction[0][0]*100, 2)))
    return flask.render_template("MelanomaImageDetector.html", message="Invalid file type")


@app.route("/SkinCancerDetector", methods=["GET", "POST"])
def skinCancerDetector():
    localizations = {'lower extremity': 0, 'foot': 1, 'hand': 2, 'abdomen': 3, 'scalp': 4, 'genital': 5, 'back': 6, 'unknown': 7, 'face': 8, 'acral': 9, 'ear': 10, 'trunk': 11, 'chest': 12, 'neck': 13, 'upper extremity': 14}

    # If image is not provided, return the rendered template
    if flask.request.method == "GET":
        return flask.render_template("SkinCancerDetector.html", message="Upload a JPG or PNG image, and answer the "
                                                                        "questions below", localizations=localizations.keys())
    # If image is provided, get the image
    image = flask.request.files["image"]
    if image.filename == '':
        return flask.render_template("SkinCancerDetector.html", message="No image selected")
    if image and allowed_file(image.filename):
        subfolder = str(random.randint(0, 999999))
        os.makedirs(app.config['UPLOAD_FOLDER'] + "/" + subfolder + "/Predict")
        path = os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], subfolder),
                            image.filename)
        image.save(path)

        metaData = [flask.request.form["age"], flask.request.form["sex"], flask.request.form["localization"]]
        metaData[0] = int(metaData[0]) if not metaData[0] else -1
        metaData[1] = {"female": 1, "male": 2, "unknown": 0}[metaData[1]]
        metaData[2] = localizations[metaData[2]]
        print(metaData)

        # Check if the image has been uploaded into the temp folder
        for x in range(900):
            if os.path.isfile(path):
                print("image uploaded")
                break
            time.sleep(0.01)
        print("image might not be uploaded")

        # Get the prediction
        prediction = modelOutputs.multiInput("SkinCancer",
                                                 os.path.join(app.config['UPLOAD_FOLDER'], subfolder) + "\\" + image.filename,
                                                 metaData,
                                                 (30, 30, 3))
        # Remove the subfolder
        shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], subfolder))
        # Return the rendered template
        indicies = np.argpartition(prediction[0], -2)
        cancerTypes = ["Actinic", "BCC", "Dermatofibroma", "Keratosis", "Melanoma", "Nevu", "Vascular"]
        predictionResult = cancerTypes[indicies[-1]] + " or " + cancerTypes[indicies[-2]]
        return flask.render_template("SkinCancerDetector.html", message="If you see this, something went wrong :/",
                                     pred=True, predictions=predictionResult)
    return flask.render_template("SkinCancerDetector.html", message="Invalid file type")


@app.route("/CovidXrayDetector", methods=["GET", "POST"])
def covidXrayDetector():
    # If image is not provided, return the rendered template
    if flask.request.method == "GET":
        return flask.render_template("CovidXrayDetector.html", message="Upload a JPG or PNG image")
    # If image is provided, get the image
    image = flask.request.files["image"]
    if image.filename == '':
        return flask.render_template("CovidXrayDetector.html", message="No image selected")
    if image and allowed_file(image.filename):
        subfolder = str(random.randint(0, 999999))
        os.makedirs(app.config['UPLOAD_FOLDER'] + "/" + subfolder + "/Predict")
        path = os.path.join(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], subfolder), "Predict"),
                            image.filename)
        image.save(path)

        # Check if the image has been uploaded into the temp folder
        for x in range(300):
            if os.path.isfile(path):
                break
            time.sleep(0.01)

        # Get the image prediction
        prediction = modelOutputs.getOutputImage("CovidXray",
                                                 os.path.join(app.config['UPLOAD_FOLDER'], subfolder) + "\\")
        # Remove the subfolder
        shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], subfolder))
        # Return the rendered template
        return flask.render_template("CovidXrayDetector.html", message="If you see this, something went wrong :/",
                                     pred=True, predictions=(round(100 - prediction[0][0] * 100, 2)))
    return flask.render_template("CovidXrayDetector.html", message="Invalid file type")


@app.route("/credits", methods=["GET", "POST"])
def credit():
    return flask.render_template("Credits.html")


@app.route("/about", methods=["GET", "POST"])
def about():
    return flask.render_template("About.html")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ["jpg", "png", "jpeg"]


# Run the flask app
if __name__ == "__main__":
    app.run(debug=True)
