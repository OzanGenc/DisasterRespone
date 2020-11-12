# Disaster Response Pipeline Project

In this project, I used real messages that sent during natural disasters. These messages were collected from social media and disaster response organizations by Figure Eight company. 

Two csv files of disaster_categories.csv and disaster_messages.csv are read, cleaned and saved to a database by running process.py
Messages were vectorized by applying TF-IDF. Multioutput RandomForestClassifier model is trained on vectorized data.  

Multiple outputs to predict are;

         related
         request
2                      offer
3                aid_related
4               medical_help
5           medical_products
6          search_and_rescue
7                   security
8                   military
9                child_alone
10                     water
11                      food
12                   shelter
13                  clothing
14                     money
15            missing_people
16                  refugees
17                     death
18                 other_aid
19    infrastructure_related
20                 transport
21                 buildings
22               electricity
23                     tools
24                 hospitals
25                     shops
26               aid_centers
27      other_infrastructure
28           weather_related
29                    floods
30                     storm
31                      fire
32                earthquake
33                      cold
34             other_weather
35             direct_report





### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
