import pandas as pd
import numpy as np
import json
import pymongo

def load_dataset_into_mongodb(json_file, client_url, db_name, collection_name):
    """
        Loads the dataset from the json file and stores it into a MongoDB database.
        
        Parameters:
        json_file (str): Path to the JSON file.
        client_url (str): MongoDB client URL.
        db_name (str): Name of the database.
        collection_name (str): Name of the collection.
        
    """

    # Open the JSON file and load the data into a dictionary
    with open(json_file, encoding="utf8") as data_file:    
        data_dict = json.load(data_file)
    
    # Parse the relevant data from the dictionary
    relevant_data = parse_relevant_data(data_dict)
    
    # Create a local database
    client = pymongo.MongoClient(client_url)
    db = client[db_name]
    collection = db[collection_name]
    
    # If the database exists, create it. Otherwise, leave it alone    
    if collection_name in client.list_database_names():
        print("The database exists.")
    else:
        collection.insert_many(relevant_data)
        
        print("mongodb thing: ", client.list_database_names())
        
    client.close()

def parse_relevant_data(data):
    """
        Parses the relevant data from the JSON file and returns a list of dictionaries.
        
        Parameters:
        data (dict): The JSON data loaded into a dictionary.
        
        Returns:
        list: A list of dictionaries containing the relevant data.
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
            'supported_languages': game_data['supported_languages'], 
            'full_audio_languages': game_data['full_audio_languages'], 
            'developers': game_data['developers'], 
            'publishers': game_data['publishers'], 
            'categories': game_data['categories'], 
            'genres': game_data['genres'], 
            'tags': game_data['tags'], 
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
            'tags': game_data['tags'] 
        })
        
    return relevant_data

def pull_collection_from_mongodb(client_url, db_name, collection_name):
    """
    Pulls the collection from the MongoDB database and returns it as a DataFrame.

    Args:
        client_url (string): The URL of the MongoDB client
        db_name (string): The name of the database
        collection_name (string): The name of the collection

    Returns:
        _type_: _description_
    """
    
    # Access the local database
    client = pymongo.MongoClient(client_url)
    db = client[db_name]
    collection = db[collection_name]
    
    # If the database exists, pull the document  
    if collection_name in client.list_database_names():
        document = db.games
    else:
        document = None
        
    client.close()
    
    return document


def main():
    DATA_FILE = "games.json"
    CLIENT_URL = "mongodb://localhost:27017/"
    DB_NAME = "games_database"
    COLLECTION_NAME = "games"
    
    # Load the dataset into MongoDB
    load_dataset_into_mongodb(DATA_FILE, CLIENT_URL, DB_NAME, COLLECTION_NAME)
    
    data = pd.DataFrame(pull_collection_from_mongodb(CLIENT_URL, DB_NAME, COLLECTION_NAME))
    
    # Process Data Here
    print(data.head())

if __name__ == "__main__":
        main()