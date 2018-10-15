import json
import pandas as pd
import requests
import pymongo
from flask import Flask
from flask import request
from flask_restplus import Resource, Api
from flask_restplus import fields
from flask_restplus import inputs
from flask_restplus import reqparse


from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle





#print(df_app)


# set dataframe
#df_app.set_index('Installs', inplace=True)
#df_app.reset_index(level=0, inplace=True)
#df_app.index.name='Sequence'
#df_app.index=df_app.index.map(str)
#df_app.to_csv('cleaned.csv')
#Genres = 4
#Content = 1
#Price = 2.3
#Type = 1
#Size = 23
#Category = 1
#form an input list


app = Flask(__name__)
api = Api(app,
          default="App prediction",  # Default namespace
          title="App_prediction",  # Documentation Title
          description="This is just a simple example to show how publish data as a service.")# Documentation Description

@api.route('/application_predict/<int:Category>/<int:Genres>')
@api.param('Category', 'Which type of category do you want to create')
@api.param('Size', 'What is the size of this application?')
@api.param('Type', 'Free or paid')
@api.param('Price', 'How much does it cost')
@api.param('Content', 'Age range')
@api.param('Genres', 'sub-type for this application')
class prediction(Resource):
    @api.response(200, 'Successful')
    @api.response(400, 'Validation Error')
    @api.response(404, 'Collection was not found')
    @api.doc(description="Predicting the amount of installation of provided application with corresponding data")
    def get(self, Category, Genres):
        def cleansing(file_name):
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            pd.set_option('max_colwidth',100)
            pd.set_option('display.width',1000)

            dic1 = {'ART_AND_DESIGN': 0, 'AUTO_AND_VEHICLES': 1, 'BEAUTY': 2, 'BOOKS_AND_REFERENCE': 3, 'BUSINESS': 4, 'COMICS': 5, 'COMMUNICATION': 6, 'DATING': 7, 'EDUCATION': 8, 'ENTERTAINMENT': 9, 'EVENTS': 10, 'FINANCE': 11, 'FOOD_AND_DRINK': 12, 'HEALTH_AND_FITNESS': 13, 'HOUSE_AND_HOME': 14, 'LIBRARIES_AND_DEMO': 15, 'LIFESTYLE': 16, 'GAME': 17, 'FAMILY': 18, 'MEDICAL': 19, 'SOCIAL': 20, 'SHOPPING': 21, 'PHOTOGRAPHY': 22, 'SPORTS': 23, 'TRAVEL_AND_LOCAL': 24, 'TOOLS': 25, 'PERSONALIZATION': 26, 'PRODUCTIVITY': 27, 'PARENTING': 28, 'WEATHER': 29, 'VIDEO_PLAYERS': 30, 'NEWS_AND_MAGAZINES': 31, 'MAPS_AND_NAVIGATION': 32}
            dic2 = {'Art & Design;Pretend Play': 1, 'Art & Design;Creativity': 2, 'Art & Design;Action & Adventure': 3, 'Auto & Vehicles': 4, 'Beauty': 5, 'Business': 7, 'Comics;Creativity': 9, 'Communication': 10, 'Dating': 11, 'Education;Education': 12, 'Education;Creativity': 14, 'Education;Music & Video': 15, 'Education;Action & Adventure': 16, 'Education;Pretend Play': 17, 'Education;Brain Games': 18, 'Entertainment;Music & Video': 20, 'Entertainment;Brain Games': 21, 'Entertainment;Creativity': 22, 'Events': 23, 'Finance': 24, 'Food & Drink': 25, 'Health & Fitness': 26, 'House & Home': 27, 'Libraries & Demo': 28, 'Lifestyle;Pretend Play': 30, 'Adventure;Action & Adventure': 31, 'Casual;Pretend Play': 35, 'Strategy': 37, 'Sports': 39, 'Word': 41, 'Casual;Creativity': 43, 'Casual;Action & Adventure': 44, 'Board': 47, 'Trivia': 48, 'Simulation;Education': 50, 'Action;Action & Adventure': 51, 'Casual;Brain Games': 52, 'Simulation;Action & Adventure': 53, 'Educational;Creativity': 54, 'Puzzle;Brain Games': 55, 'Educational;Education': 56, 'Card;Brain Games': 57, 'Educational;Brain Games': 58, 'Educational;Pretend Play': 59, 'Entertainment;Education': 60, 'Casual;Education': 61, 'Music;Music & Video': 62, 'Racing;Action & Adventure': 63, 'Arcade;Pretend Play': 64, 'Role Playing;Action & Adventure': 65, 'Simulation;Pretend Play': 66, 'Puzzle;Creativity': 67, 'Sports;Action & Adventure': 68, 'Educational;Action & Adventure': 69, 'Arcade;Action & Adventure': 70, 'Entertainment;Action & Adventure': 71, 'Puzzle;Action & Adventure': 72, 'Strategy;Action & Adventure': 73, 'Music & Audio;Music & Video': 74, 'Health & Fitness;Education': 75, 'Adventure;Education': 76, 'Board;Brain Games': 77, 'Board;Action & Adventure': 78, 'Board;Pretend Play': 79, 'Casual;Music & Video': 80, 'Role Playing;Pretend Play': 81, 'Entertainment;Pretend Play': 82, 'Video Players & Editors;Creativity': 83, 'Card;Action & Adventure': 84, 'Medical': 85, 'Social': 86, 'Shopping': 87, 'Photography': 88, 'Travel & Local': 89, 'Travel & Local;Action & Adventure': 90, 'Tools': 91, 'Tools;Education': 92, 'Personalization': 93, 'Productivity': 94, 'Parenting': 95, 'Parenting;Music & Video': 96, 'Parenting;Education': 97, 'Parenting;Brain Games': 98, 'Weather': 99, 'Video Players & Editors': 100, 'Video Players & Editors;Music & Video': 101, 'News & Magazines': 102, 'Maps & Navigation': 103, 'Health & Fitness;Action & Adventure': 104, 'Educational': 105, 'Casino': 106, 'Adventure;Brain Games': 107, 'Trivia;Education': 108, 'Lifestyle;Education': 109, 'Books & Reference;Creativity': 110, 'Books & Reference;Education': 111, 'Puzzle;Education': 112, 'Role Playing;Education': 113, 'Role Playing;Brain Games': 114, 'Strategy;Education': 115, 'Racing;Pretend Play': 116, 'Communication;Creativity': 117, 'February 11, 2018': 118, 'Strategy;Creativity': 119,\
            'Art & Design': 0, 'Comics': 8, 'Education': 13, 'Entertainment': 19, 'Music': 40, 'Casual': 33, 'Simulation': 45, 'Puzzle': 38, 'Books & Reference': 6, 'Lifestyle': 29, 'Card': 34, 'Arcade': 32, 'Racing': 42, 'Action': 36, 'Adventure': 46, 'Role Playing': 49}

            # read original csv file
            csv_file = str(file_name)
            df_app = pd.read_csv(csv_file)
            columns_to_drop = ['Reviews', 'Type', 'Last Updated','Current Ver','Android Ver']


            # deal with file
            df_app.drop(columns_to_drop, inplace=True, axis=1)
            df_app.fillna(0,inplace=True)
            for i in dic1:
                df_app['Category'] = df_app['Category'].apply(lambda x: x.replace(i, str(dic1[i])))
            df_app['Size'] = df_app['Size'].apply(lambda x:  int(float(x[:-1])*1000) if 'M' in x  else int(float(x[:-1])) if 'k' in x else -1)
            df_app.drop(df_app.loc[df_app['Size']==-1].index, inplace=True)
            df_app['Price'] = df_app['Price'].apply(lambda x: float(x[1:]) if '$' in x else float(x) if x=='0' else -1)
            df_app.drop(df_app.loc[df_app['Price']==-1].index, inplace=True)
            df_app['Content Rating'] = df_app['Content Rating'].apply(lambda x: 1.0 if x=='Everyone' else 1.5 if x=='Everyone 10+' else 2.0 if x=='Teen' else 3.0 if x=='Mature 17+' else -1)
            df_app.drop(df_app.loc[df_app['Content Rating']==-1].index, inplace=True)
            df_app['Genres'] = df_app['Genres'].apply(lambda x: dic2[x])
            return df_app      

        def data_predict(df, inputlist):
            x_train = df.drop(['App','Rating','Installs'], axis=1).values
            y_train = df['Installs'].values
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(x_train, y_train)
            predictions = knn.predict(inputlist)
            return predictions[0]

        def sim_app(df,inputlist, predict_install):
            sim_app = ''
            df.set_index('Installs', inplace=True)
            df.reset_index(level=0, inplace=True)
            df.index.name='Sequence'
            df.index=df.index.map(str)
            #print(df.index[0])
            for i in range(len(df.index)):
                
                if(inputlist[0][0]==df.at[df.index[i],'Category']):
                    if(inputlist[0][4]==df.at[df.index[i],'Genres']):
                        if(predict_install==df.at[df.index[i],'Installs']):
                            sim_app = df.at[df.index[i],'App']
                            return sim_app

        Size = float(request.args.get('Size'))
        Type = int(request.args.get('Type'))
        Price = float(request.args.get('Price'))
        Content = int(request.args.get('Content'))
        inputlist = [[Category, Size, Price, Content, Genres]]
        

        df = cleansing('googleplaystore.csv')
        #input the dataframe and input list to the function
        #the output is the prediction
        predict_install = data_predict(df, inputlist)
        result_sim_app = sim_app(df, inputlist, predict_install)       

        return {"predict_install": "{}".format(str(predict_install)),
                "result_sim_app": "{}".format(str(result_sim_app)),
                }, 200


if __name__ == '__main__':
    app.run(debug=True)
