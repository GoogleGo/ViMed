import pandas
import tensorflow as tf
import numpy as np


# Class to initialize the models and output the results
class ModelOutputs:
    def __init__(self):
        self.models = {}

    # Add a model to the dictionary, returns 0 if successful, 1 if not
    def createModel(self, modelName: str = None, modelDir: str = ""):
        if modelName is None or modelDir is None:
            return 1
        if ".h5" not in modelDir:
            return 1

        model = tf.keras.models.load_model(modelDir)
        self.models[modelName] = model
        return 0

    def getOutputImage(self, modelName: str = "", imageDirectory: str = "", targetSize: tuple = (300, 300)):
        generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        dataGen = generator.flow_from_directory(imageDirectory, target_size=targetSize, batch_size=1)
        model = self.getModel(modelName)
        return model.predict(dataGen, steps=1)

    def getOutputRegression(self, modelName: str = "", Xi: list = None):
        if Xi is None:
            return False
        return self.models[modelName].predict(Xi)

    def removeModel(self, modelName: str = ""):
        if modelName == "":
            return
        if modelName in self.models:
            del self.models[modelName]
            return
        return

    def getModel(self, modelName: str = ""):
        if modelName == "":
            return
        if modelName in self.models:
            return self.models[modelName]
        raise Exception("Model not found:", self.models)

    def pasrseInSympt(self, sympton):
        symptDict = self.hash_data(self.get_input_symptoms())
        return symptDict[sympton]

    def parseOutSympt(self, prediction):
        disDict = self.hash_data(prediction)
        return disDict.items()[prediction.argmax()][0]

    # Get the possible input symptoms from dataset.csv - Same function from the Training Scripts
    def get_input_symptoms(self):
        data = pandas.read_csv('DiseaseSymptomDataset/dataset.csv')
        # get all the values in each column but the first one
        dat = data.iloc[:, 1:].values.tolist()
        extracted = []
        for i in dat:
            for j in i:
                extracted.append(j)
        # remove spaces
        extracted = [x.strip() if type(x) is str else "NONE" for x in extracted]
        # remove duplicates
        extracted = list(set(extracted))
        extracted.remove("NONE")
        return extracted

    # Get the diseases in a list, and the corresponding symptoms in a list - Same function from the Training Scripts
    # EX: [disease1, disease2, disease3] and [[symptoms1], [symptoms2], [symptoms3]]
    def get_diseases(self):
        data = pandas.read_csv('DiseaseSymptomDataset/dataset.csv')
        # get all the rows in the dataset
        dat = data.iloc[:, :].values.tolist()
        # get the first column in the dataset
        diseases = data.iloc[:, 0].values.tolist()

        # hash the diseases
        hash_diseases = self.hash_data(diseases)
        for i in range(len(diseases)):
            diseases[i] = hash_diseases[diseases[i]]

        # remove all the spaces
        hashed = self.hash_data(self.get_input_symptoms())
        for i in range(len(dat)):
            for j in range(len(dat[i])):
                dat[i][j] = dat[i][j].strip() if type(dat[i][j]) is str else "NONE"
            # Remove the first item in the list
            dat[i].remove(dat[i][0])
            # Replace the symptoms with their corresponding hash
            for j in range(len(dat[i])):
                if dat[i][j] in hashed:
                    dat[i][j] = hashed[dat[i][j]]
                else:
                    dat[i][j] = 0
        return diseases, dat

    def hash_data(self, dat):  # Same hashing function from the Training Scripts
        data = dat.copy()
        # sort
        data.sort()
        # hash the symptoms by creating a dictionary
        hash_symptoms = {}
        if "NONE" in data:
            data.remove("NONE")
        currentIndex = 1
        for i in range(len(data)):
            if data[i] not in hash_symptoms:
                hash_symptoms[data[i]] = currentIndex
                currentIndex += 1
        hash_symptoms["NONE"] = 0
        return hash_symptoms

    def multiInput(self, modelName: str = "", Xi: str = "", Xi2: list = None, targetSize: tuple = (30, 30, 3)):
        print(Xi, Xi2)

        if Xi is None:
            return False
        image = tf.keras.preprocessing.image.load_img(Xi, target_size=targetSize)
        print(np.shape([[np.array(image)], [np.array(Xi2)]]))
        image = np.array(image).reshape((1, 30, 30, 3))
        meta = np.array(Xi2).reshape((1, 3))
        return self.models[modelName].predict_on_batch(x={"img": image, "meta": meta})

