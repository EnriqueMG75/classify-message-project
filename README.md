# Disaster Response Pipeline Project
### Project Overview:


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. To run the application you can either use the flask command or python’s -m switch with Flask. Before you can do that you need to tell your terminal the application to work with by exporting the FLASK_APP environment variable:

	$ export FLASK_APP=run.py
	$ flask run
 	* Running on http://127.0.0.1:5000/

If you are on Windows, the environment variable syntax depends on command line interpreter. On Command Prompt:

	C:\path\to\app>set FLASK_APP=run.py

And on PowerShell:

	PS C:\path\to\app> $env:FLASK_APP = python "run.py"    <--- Recommended    

(Source: https://flask.palletsprojects.com/en/1.1.x/quickstart/#a-minimal-application)
(See Flask documentation for more info)

3. Go to http://127.0.0.1:5000/
