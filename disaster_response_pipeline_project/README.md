# Disaster Response Pipeline Project

In this project, I used real messages that sent during natural disasters. These messages were collected from social media and disaster response organizations by Figure Eight company. 

Two csv files of disaster_categories.csv and disaster_messages.csv are read, cleaned and saved to a database by running process.py
Messages were vectorized by applying TF-IDF. Multioutput RandomForestClassifier model is trained on vectorized data.  

Multiple outputs to predict are;

         related
         request
                     offer
              aid_related
               medical_help
           medical_products
          search_and_rescue
                   security
                   military
                child_alone
                     water
                      food
                   shelter
                  clothing
                     money
            missing_people
                  refugees
                     death
                 other_aid
    infrastructure_related
                 transport
                 buildings
               electricity
                     tools
                 hospitals
                     shops
               aid_centers
      other_infrastructure
           weather_related
                    floods
                     storm
                      fire
                earthquake
                      cold
             other_weather
             direct_report





### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
