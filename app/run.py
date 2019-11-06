import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import re
from collections import Counter


app = Flask(__name__)

def load_data():
    # load data from database
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('MessageNewTable', engine)
    
    # Take a sample to run faster:
    df = df.sample(n=1000, random_state=42)
    
    # Define feature and target variables X and Y and category names
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    
    return X, Y

def tokenize(text):
    # remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def count_words():
    '''
    Function to count words. It is useful in the 'words more repeated plot' 
    '''
    X, Y = load_data()
    unique = []
    for message in X:
        tokens = tokenize(message)
     
        if tokens not in unique:
            unique.append(tokens)
  
    
    flat_list = []
    for sublist in unique:
        for item in sublist:
            flat_list.append(item)
        
    counter = Counter(flat_list)
    labels, values = zip(*counter.items())
    
    return labels, values

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageNewTable', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # Number of messages by class
    Y = df.iloc[:,4:]
    message_classes = Y.sum()
    classes_names = list(message_classes.index)
    # Words more repeated
    labels, values = count_words()
    # create visuals
    
    graphs = [
      
        {
            'data': [
                Bar(
                    x=classes_names,
                    y=message_classes
                )
            ],

            'layout': {
                'title': 'Number of Messages by Class',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Classes"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=labels,
                    y=values
                )
            ],

            'layout': {
                'title': 'Words more repeated',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },
          {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()