# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """
    This function loads the data set from the path shared 
    in the argument. mergers both the datasets 
    
    parameters : file path
    return: dataframe
    
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, how='left', on="id")
    
    return df


def clean_data(df):
    """
    this function cleans the data by removing unwanted infomration 
    on the data, splitting and copying only the numeric value necessary 
    for the analysis, removes duplicated value
    and adds the approprate column, back to the dataframe.
    param: dataframe
    
    return: cleaned df

    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.map(lambda x: str(x)[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:].astype(str)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop('categories', inplace=True, axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],  axis=1, join='inner', sort = False) 
    
    #find duplicates
    df.duplicated(subset = ['id', 'message']).sum()
    # drop duplicates
    df.drop_duplicates(subset = ['id', 'message'], inplace =True)
    
    #remove unwanted values
    df.related.replace(2,1,inplace=True)
    print(df.head(5))
    
    return df        
        

def save_data(df, database_filename):
    """
    saves the cleaned data to the sql database
    
    param:dataframe to load, and database filename
    
    return: none

    """  
    engine = create_engine('sqlite:///'+ database_filename)
    print(df.head())
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)
    
    
    

def main():
    """
    performs the full operation of ETL.
    
    param: the user to pass the path for the datasets, and the path for database
    
    return:none

    """
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