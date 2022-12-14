# ViMed
### Vision Medical - Online Diagnosis

## Installing and Running the app
Download the code in a .zip format, and unzip it. You can do it through the greed "Code" button (desktop).

Use any code editor, or Python's native excecutor. Make sure to download all the packages below:
- flask
- numpy
- shutil
- tensorflow and it's dependencies
- pandas
- PIL  

Preferably, use PyCharm so the packages will download automatically. Python 3.9+ is reccommended for this project.

### Unzipping
In the [Models](https://github.com/GoogleGo/ViMed/tree/main/Models) directory, you can notice that the files are compressed into the ".rar" format. To decompress the models, you will need to use [WinRAR](https://www.win-rar.com/start.html?&L=0). These models are compressed due to Github's CL upload limit. Once decompressed, it should be in the format ".h5".

### Running
After the program begins to run, there will be some loading time as files are imported and assets are loaded. When flask finishes, it will provide a localhost address that the local website will run on. Going to the link will bring you to the app.

## Models
The models may take some time for certain images, as Tensorflow may need to load assets or images back into memory. Depending on the computer used, the models will take different amounts of time to load the images and categorize them.
