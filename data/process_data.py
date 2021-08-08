import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories data from csv
    Input:
        messages_filepath: Path to disaster messages csv
        categories_filepath: Path to disaster categories csv
    Output:
        df: Dataframe with merged dataset
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = categories.merge(messages, how="outer", on=["id"])
    return df


def clean_data(df):
    """
    Clean dataset
    Input:
        df: Input dataframe
    Output:
        df: Cleaned dataframe
    """
    # Split categories into separate columns
    categories = df["categories"].str.split(";", expand=True)
    category_colnames = [x.split("-")[0] for x in categories.iloc[0]]
    # Rename columns
    categories.columns = category_colnames
    # Modify records to be bool
    categories[category_colnames] = categories[category_colnames].applymap(lambda x : int(x.split("-")[1])==1)
    # Update dataframe
    df = df.drop("categories", axis=1)
    df = pd.concat([df, categories], axis=1)
    # Drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Save dataframe to SQL database
    Input:
        df: Dataframe
        database_filename: Path to SQL database
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(f'disaster_data', engine, index=False)    


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