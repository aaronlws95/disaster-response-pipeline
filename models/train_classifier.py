import sys
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer class that checks if the sentence starts with a verb
    """
    def starting_verb(self, text):
        """
        Checks if the starting word of text is a verb
        Input:
            text: Input text
        Output:
            True if starting word in text is a verb else False
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """
        Apply transform. Overloaded function.
        Input:
            X: input text data
        Output:
            Input text data transformed by `starting_verb`
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """
    Load data from SQL database
    Input:
        database_filepath: Path to SQL database
    Output:
        X: Messages
        y: Category class label
        y.columns: Category names
    """
    df = pd.read_sql_table('disaster_data', f'sqlite:///{database_filepath}')  
    X = df["message"]
    y = df.drop(["message", "id", "original", "genre"], axis=1) 
    return X, y, y.columns


def tokenize(text):
    """
    Clean and tokenize a text input
    Input:
        text: Input text
    Output:
        clean_tokens: Cleaned and tokenized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Unify URLs with a placeholder
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Split text into words
    tokens = nltk.word_tokenize(text)
    
    # Group inflected words, remove whitespace, transform all to lowercase
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(x).lower().strip() for x in tokens]
    
    return clean_tokens


def build_model(grid_search=False):
    """
    Build the ML pipeline
    Output:
        pipeline: ML pipeline
    """
    classifier = RandomForestClassifier()

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )    
    }

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
        
        ('clf', MultiOutputClassifier(classifier))
    ])

    return GridSearchCV(pipeline, param_grid=parameters) if grid_search else pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate the model and report f1 score, precision, recall
    Input:
        model: ML model
        X_test: Test split messages
        y_test: Test split category class label
        category_names: Category labels
    """
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns = category_names)
    accuracy = []
    for col in category_names:
        print(f"Category: {col}")
        report = classification_report(y_test[col],y_pred[col], output_dict=False)
        print(report)
    

def save_model(model, model_filepath):
    """
    Save model to pickel file
    Input:
        model: ML model
        model_filepath: Path to model pickel file
    """
    joblib.dump(model, f"{model_filepath}")


def main():
    parser = argparse.ArgumentParser(description='Train and save ML classifier')
    parser.add_argument('database_filepath', type=str, help='Path to database')
    parser.add_argument('model_filepath', type=str, help='Path to save model')
    parser.add_argument('--grid_search', action="store_true", help='Enable grid search')
    args = parser.parse_args()    

    database_filepath = args.database_filepath
    model_filepath = args.model_filepath
    
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    print('Building model...')
    model = build_model(args.grid_search)
    
    print('Training model...')
    if args.grid_search:
        print('Grid search is enabled, training will take a while')
    model.fit(X_train, Y_train)
    if args.grid_search:
        print(f"Best params: {model.best_params_}")

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    main()