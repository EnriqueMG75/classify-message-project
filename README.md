# Classifying Messages Using Machine Learning and NLP
### Project Overview:

This project is about using NLP and Machine Learning to train a model that can classify messages.
This project belong to the Udacity Data Scientist Nanodegree (www.udacity.com/course/data-scientist-nanodegree--nd025) and use disaster messages data from Figure Eight (www.figure-eight.com)

The project have 3 parts:

1. Build an ETL pipeline

Using Data Engineering, I have built an ETL pipeline where I load the data from csv files, I clean and transform  the data and I save the data to a Data Base

2. Build a ML pipeline

Using Machine Learning Engineering and NLP processing, I have built an ML pipeline where I tokenize the message and I build, train, evaluate and save the model

3. Build a Web App

Using the Web App framework Flask, I have created a web app where users can interact with the trained model. Users can input a message and the app returns the categories which belongs the message to. Also they can explore the data with some plots

### Instructions:

1. To run the application you can either use the flask command or python’s -m switch with Flask. Before you can do that you need to tell your terminal the application to work with by exporting the FLASK_APP environment variable:

	$ export FLASK_APP=run.py
	$ flask run
 	* Running on http://127.0.0.1:5000/

If you are on Windows, the environment variable syntax depends on command line interpreter. On Command Prompt:

	C:\path\to\app>set FLASK_APP=run.py

And on PowerShell:

	PS C:\path\to\app> $env:FLASK_APP = python "run.py"    <--- Recommended    

(Source: https://flask.palletsprojects.com/en/1.1.x/quickstart/#a-minimal-application)
(See Flask documentation for more info)

2. Go to http://127.0.0.1:5000/

3. Run the following commands in the project's root directory to set up your database and model.
In the case you want to use other files you can pass it using the same arguments structure 

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


