# Disaster Response Pipeline Project

This repo contains scripts for a machine learning (ML) workflow that categorizes messages into disaster respones categories. First, we clean and process the data with a extract, transform, load ([ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load)) pipeline followed by training a machine learning model to categorize the messages. Finally, we present a simple web app that runs the model and displays some visualizations.

This work was carried out as homework as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

### Setup:
Setup Python environment

```bash
# Recommended to setup a conda environment
conda create -n disaster-pipeline python=3.7 
conda activate disaster-pipeline
# Install requirements
pip install -r requirements.txt 
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - To run ML pipeline that trains the classifier and saves with parameter grid search 
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl --grid_search`

2. Run the following command in the app's directory to run your web app. (`cd app`)
    `python run.py ../data/DisasterResponse.db ../models/classifier.pkl`

3. Go to http://0.0.0.0:3001/

### Notebooks

1. `data/1 - ETL Pipeline Preparation`
    * Rudimentary version of `process_data.py` in notebook form
2. `data/2 - Exploration Data Analysis`
    * Analysis of data produced by `process_data.py`
3. `models/1 - ML Pipeline Preparation`
    * Rudimentary version of `train_classifier.py` in notebook form
