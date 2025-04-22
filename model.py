""" Import Statements """
import pandas as pd
import numpy as np
import seaborn as sns
import json
import pymongo
import matplotlib.pyplot as plt



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
        operating_system = []
        if game_data["windows"]: operating_system.append("windows")
        if game_data["linux"]: operating_system.append("linux")
        if game_data["mac"]: operating_system.append("mac")
        
        relevant_data.append({
            'name': game_data['name'],
            'required_age': game_data['required_age'],
            'price': game_data['price'],
            'metacritic_score': game_data['metacritic_score'],
            'recommendations': game_data['recommendations'],
            'supported_languages': game_data['supported_languages'], 
            'developers': game_data['developers'], 
            'publishers': game_data['publishers'],
            'categories': game_data['categories'], 
            'genres': game_data['genres'], 
            'tags': game_data['tags'], 
            'score_rank': game_data['score_rank'],
            'positive_reviews': game_data['positive'],
            'negative_reviews': game_data['negative'],
            'estimated_owners': game_data['estimated_owners'],
            'average_playtime_forever': game_data['average_playtime_forever'],
            'average_playtime_2weeks': game_data['average_playtime_2weeks'],
            'median_playtime_forever': game_data['median_playtime_forever'],
            'median_playtime_2weeks': game_data['median_playtime_2weeks'],
            'peak_ccu': game_data['peak_ccu'],
            'operating_system': operating_system
        })
        
    return relevant_data

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
    
    print("Database Names: ", client.list_database_names())
    print("games_database collection Names: ", db.list_collection_names())
    
    # If the database exists, create it. Otherwise, leave it alone    
    if collection.count_documents({}) > 0:
        collection.delete_many({})
        collection.insert_many(relevant_data)
        print("The collection already has data.")
    else:
        collection.insert_many(relevant_data)
        print("The collection was populated with data.")
    
    client.close()
    
def pull_dataframe_from_mongodb(client_url, db_name, collection_name):
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
    
    # If the database exists, pull the collection. Otherwise, leave it alone
    if collection_name in db.list_collection_names():
        print("The collection exists for retrieving data.")
        collection_data = collection.find()
    else:
        print("The collection does not exist.")
        collection_data = None
    
    # Convert the collection data to a DataFrame
    df = pd.DataFrame(list(collection_data))
    
    # Drop the '_id' column if it exists
    if '_id' in df.columns:
        df.drop(columns=['_id'], inplace=True)
    
    client.close()
    
    return df

def plot_scatterplot(data, x_column, y_column):
    new_df = data[[x_column, y_column]]

    plt.scatter(new_df[x_column], new_df[y_column])

    meanx = new_df[x_column].mean()
    meany = new_df[y_column].mean()
    plt.axvline(meanx, color='r', linestyle='--', label='Mean of X')
    plt.axhline(meany, color='g', linestyle='--', label='Mean of Y')

    plt.xlabel(x_column)
    plt.ylabel('Positive Reviews')
    plt.title(f'Scatter plot of {x_column} vs {y_column} (All Values)')

    # Add legend
    plt.legend()
    plt.show()
    plt.close()

def plot_scatterplot_without_extremes(data, x_column, y_column, x_extreme, y_extreme):
    new_df = data[(data[x_column] < x_extreme) & (data[y_column] < y_extreme)]

    plt.scatter(new_df[x_column], new_df[y_column])

    meanx = new_df[x_column].mean()
    meany = new_df[y_column].mean()
    plt.axvline(meanx, color='r', linestyle='--', label='Mean of X')
    plt.axhline(meany, color='g', linestyle='--', label='Mean of Y')

    plt.xlabel(x_column)
    plt.ylabel('Positive Reviews')
    plt.title(f'Scatter plot of {x_column} vs {y_column} (without Extremes)\n{x_column} < {x_extreme} \n {y_column} < {y_extreme}')

    # Add legend
    plt.legend()
    plt.show()
    plt.close()

from sklearn.preprocessing import MultiLabelBinarizer
def one_hot_encode_tags(data, tags_column, dependant_variable='positive_reviews', batch_size=100, plot=False):
    mlb = MultiLabelBinarizer()
    
    # List of all unique tags
    unique_tags = data[tags_column].explode().unique()

    # Split unique tags into manageable chunks
    tag_batches = [unique_tags[i:i + batch_size] for i in range(0, len(unique_tags), batch_size)]
    
    # Initialize a list to hold the batch-wise encoded DataFrames
    all_encoded_tags = []

    for batch in tag_batches:
        # Filter the dataset to only contain rows where tags are in the current batch
        tags_subset = data[data[tags_column].apply(lambda x: any(tag in batch for tag in x))]

        # Apply MultiLabelBinarizer for the current batch of tags
        tags_encoded = mlb.fit_transform(tags_subset[tags_column])
        tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_)

        # Combine with the dependent variable (positive_reviews)
        data_encoded = pd.concat([tags_df, tags_subset[[dependant_variable]]], axis=1)

        # Append the batch's encoded DataFrame to the list
        all_encoded_tags.append(data_encoded)
    
    # Combine all the batches into one final DataFrame
    final_encoded_tags_df = pd.concat(all_encoded_tags, axis=0)
    
    # Compute correlation with the dependent variable
    correlations = final_encoded_tags_df.corr()[dependant_variable].drop(dependant_variable)
    
    tags_encoded = mlb.fit_transform(data[tags_column])

    tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_)

    # Combine with positive reviews
    data_encoded = pd.concat([data[[dependant_variable]], tags_df], axis=1)

    # Compute correlation with positive reviews
    correlations = data_encoded.corr()[dependant_variable].drop(dependant_variable)

    print(f"\nCorrelation of {tags_column} with Positive Reviews:\n", correlations.head().to_string())

    if(plot):
        # Sort and plot
        plt.figure(figsize=(10, 75))
        correlations.sort_values().plot(kind='barh', color='skyblue')
        plt.title(f'Correlation of {tags_column} with Positive Reviews')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel(tags_column)
        plt.grid(True)
        plt.show()
    
    return tags_df


# Main function to load the dataset into MongoDB and pull it as a DataFrame

DATA_FILE = "games.json"
CLIENT_URL = "mongodb://localhost:27017/"
DB_NAME = "games_database"
COLLECTION_NAME = "games"

# Load the dataset into MongoDB
all_data = pull_dataframe_from_mongodb(CLIENT_URL, DB_NAME, COLLECTION_NAME)

data = all_data.sample(n=500)

# Process Data Here

# Changes the required_age column to a boolean value if under 18 or not
data['required_age'] = np.where(data['required_age'] >= 18, 1, 0)
data.rename(columns={'required_age': 'over_18_required'}, inplace=True)

print("\n", data.head())

# categories -> binarized
# genres -> binarized
# publisher -> binarized
# developers -> binarized
# supported_languages -> number of languages
# full_audio_languages -> 0 or 1
# required_age -> 0 or 1

# could do: tags

# List of columns you want to process
categorical_columns = [
    'categories',
    'genres',
    'publishers',
    'developers',
    'supported_languages',
]

# Store all encoded DataFrames in a list
encoded_dfs = []

for column in categorical_columns:
    print(f"Encoding column: {column}")
    encoded_df = one_hot_encode_tags(data, column, dependant_variable='positive_reviews')  # your existing function
    encoded_dfs.append(encoded_df)

# Combine them all into a single DataFrame (side-by-side)
combined_encoded_df = pd.concat(encoded_dfs, axis=1)

# Drop any duplicated columns if there are overlaps
combined_encoded_df = combined_encoded_df.loc[:, ~combined_encoded_df.columns.duplicated()]

print(combined_encoded_df.head())

X = combined_encoded_df
y = data['positive_reviews']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)



# Evaluate the model performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae} / {len(X_train)}   {mae/len(X_train)}")

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance on Test Set:")
print(f"  Mean Absolute Error (MAE): {mae:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  R² Score: {r2:.2f}")

importances = model.feature_importances_
features = X.columns


# Create a DataFrame for sorting and plotting
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Top 20 features
print("\nTop 20 Most Important Features:")
print(feature_importances.head(20).to_string(index=False))

# Plot
plt.figure(figsize=(10, 8))
feature_importances.head(20).plot(kind='barh', x='Feature', y='Importance', legend=False, color='skyblue')
plt.title('Top 20 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Predicted versus actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3, edgecolors='w')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Positive Reviews')
plt.ylabel('Predicted Positive Reviews')
plt.title('Predicted vs. Actual')
plt.grid(True)
plt.tight_layout()
plt.show()


# Distribution of Errors
errors = y_test - y_pred
plt.figure(figsize=(8, 4))
plt.hist(errors, bins=50, color='salmon', edgecolor='black')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()


# Some predicted versus actual
comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print("\nSample Predictions:")
print(comparison_df.head(10).to_string(index=False))

# Correlations
correlations = X.corrwith(y)
print("\nTop correlations with target:")
print(correlations.sort_values(ascending=False).head(10).to_string())