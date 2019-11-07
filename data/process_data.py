

# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load data from csv files
    Args: csv filepaths
    Returns: Pandas DataFrame with the two files merged
    '''
    messages = pd.read_csv('disaster_messages.csv')
    categories = pd.read_csv('disaster_categories.csv')
    # merge datasets
    df = messages.merge(categories, how='outer', on='id')

    return df

      


def clean_data(df):
    '''
    Several transforming and cleaning processes
    Arg: Dataframe to clean
    Return: Cleaned DataFrame
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # extract a list of new column names for categories
    row_split = row.str.split('-') 
    category_colnames = [x[0] for x in row_split] 
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1.
    for i in category_colnames:
        # set each value to be the last character of the string
        categories[i] = categories[i].str.split('-').str.get(1)
        # Change to integer type
        categories[i] = pd.to_numeric(categories[i], downcast='integer')
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df_concat = pd.concat([df, categories],axis=1, sort=False)
    # drop duplicates
    df_concat.drop_duplicates(inplace=True)
    df = df_concat

    return df

def save_data(df, database_filename):
    '''
    Save data to a SQlite DataBase
    Args: DataFrame
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessageNewTable2', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()