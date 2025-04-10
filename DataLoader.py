import pandas as pd
import numpy as np
import json
import pymongo

def load_dataset():
    """
        Loads the dataset from games.csv
    """

    with open('games.json', encoding="utf8") as data_file:    
        y = json.load(data_file)

    # the result is a Python dictionary:
    #for key in y:
        #print(y[key]["name"])
    
    relevant_data = parse_relevant_data(y)
    
    df = pd.DataFrame(relevant_data)
    
    print(df)

    return df

def parse_relevant_data(data):
    """
        Parses the relevant data from the JSON file.
    """
    
    relevant_data = []
    
    for game in data:
        game_data = data[game]
        relevant_data.append({
            'name': game_data['name'],
            'release_date': game_data['release_date'],
            'required_age': game_data['required_age'],
            'price': game_data['price'],
            'dlc_count': game_data['dlc_count'],
            'detailed_description': game_data['detailed_description'],
            'about_the_game': game_data['about_the_game'],
            'short_description': game_data['short_description'],
            'windows': game_data['windows'],
            'mac': game_data['mac'],
            'linux': game_data['linux'],
            'metacritic_score': game_data['metacritic_score'],
            'achievements': game_data['achievements'],
            'recommendations': game_data['recommendations'],
            'supported_languages': game_data['supported_languages'], #could be a list
            'full_audio_languages': game_data['full_audio_languages'], #could be a list
            'developers': game_data['developers'], #could be a list
            'publishers': game_data['publishers'], #could be a list
            'categories': game_data['categories'], #could be a list
            'genres': game_data['genres'], #could be a list
            'tags': game_data['tags'], #could be a list
            'user_score': game_data['user_score'],
            'score_rank': game_data['score_rank'],
            'positive_reviews': game_data['positive'],
            'negative_reviews': game_data['negative'],
            'estimated_owners': game_data['estimated_owners'],
            'average_playtime_forever': game_data['average_playtime_forever'],
            'average_playtime_2weeks': game_data['average_playtime_2weeks'],
            'median_playtime_forever': game_data['median_playtime_forever'],
            'median_playtime_2weeks': game_data['median_playtime_2weeks'],
            'peak_ccu': game_data['peak_ccu'],
            'tags': game_data['tags'] #could be a list
        })
        
    return relevant_data

def storeData(data_frame):
    """
        Stores the data in a MongoDB database.
    """
    
    # Create a local database
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["games_database"]
    collection = db["games"]
    
    # If the database exists, create it. Otherwise, leave it alone    
    if "games" in client.list_database_names():
        print("The database exists.")
    else:
        data_dict = data_frame.to_dict(orient='records')
        
        collection.insert_many(data_dict)
        
        print("mongodb thing: ", client.list_database_names())
        
    client.close()
    

def main():
    data = load_dataset()
    
    storeData(data)
    
    print(data.head())

if __name__ == "__main__":
        main()