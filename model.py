""" Import Statements """
import pandas as pd
import numpy as np
import seaborn as sns
import json
import pymongo
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
        
        if game_data['supported_languages']: num_languages = len(game_data['supported_languages'])
        if len(game_data['full_audio_languages']) > 0:
            full_audio = 1
        else:
            full_audio = 0
        
        relevant_data.append({
            'name': game_data['name'],
            'required_age': game_data['required_age'],
            'price': game_data['price'],
            'recommendations': game_data['recommendations'],
            'supported_languages': num_languages, 
            'developers': game_data['developers'], 
            'publishers': game_data['publishers'],
            'categories': game_data['categories'], 
            'full_audio': full_audio,
            'genres': game_data['genres'], 
            'tags': game_data['tags'], 
            'score_rank': game_data['score_rank'],
            'net_sentiment': game_data['positive'] - game_data['negative'],
            'positivity_ratio': game_data['positive'] / (game_data['positive'] + game_data['negative'] + 1),
            'estimated_owners': game_data['estimated_owners'],
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
        
    # If the database exists, create it. Otherwise, leave it alone    
    if collection.count_documents({}) > 0:
        print("The collection already has data.")
        collection.delete_many({})
        collection.insert_many(relevant_data)

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
    
    print("pull_dataframe...", df.columns)
    
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
def one_hot_encode_tags(data, tags_column, dependant_variable='net_sentiment', batch_size=100, plot=False):    
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
        
    tags_encoded = mlb.fit_transform(data[tags_column])

    tags_df = pd.DataFrame(tags_encoded, columns=mlb.classes_)

    # Combine with positive reviews
    data_encoded = pd.concat([data[[dependant_variable]], tags_df], axis=1)
    
    return data_encoded


# Function to compute the mean of the range
def convert_range_to_mean(range_str):
    lower, upper = range_str.split('-')
    return (int(lower) + int(upper)) / 2


# Main function to load the dataset into MongoDB and pull it as a DataFrame
DATA_FILE = "games.json"
CLIENT_URL = "mongodb://localhost:27017/"
DB_NAME = "games_database"
COLLECTION_NAME = "games"

load_dataset_into_mongodb(DATA_FILE,CLIENT_URL,DB_NAME,COLLECTION_NAME)

# Load the dataset into MongoDB
all_data = pull_dataframe_from_mongodb(CLIENT_URL, DB_NAME, COLLECTION_NAME)
data = all_data.sample(n=100)


# Changes the required_age column to a boolean value if under 18 or not
data['required_age'] = np.where(data['required_age'] >= 18, 1, 0)

data['estimated_owners'] = data['estimated_owners'].apply(convert_range_to_mean)

print("\n", data.head())
print("---", data.columns)

# List of columns you want to process
columns_to_encode = [
    'categories',
    'genres',
    'tags',
    'publishers',
    'developers',
    'operating_system',
]

binary_columns = ['full_audio', 'required_age']

numerical_columns = ['supported_languages']

# Store all encoded DataFrames in a list
encoded_dfs = []

for column in columns_to_encode:
    print(f"Encoding column: {column}")
    encoded_df = one_hot_encode_tags(data, column) 
    encoded_df = encoded_df.set_index(data.index)  # force alignment
    encoded_dfs.append(encoded_df)

# Combine them all into a single DataFrame (side-by-side)
combined_encoded_df = pd.concat(encoded_dfs, axis=1)

print(":::", combined_encoded_df)
print(":::", pd.concat([ data[binary_columns].copy()], axis=1))
print(":::", pd.concat([ data[numerical_columns].copy()], axis=1))

# Drop any duplicated columns if there are overlaps
combined_encoded_df = combined_encoded_df.loc[:, ~combined_encoded_df.columns.duplicated()]

binary_df = data[binary_columns].copy()
numerical_df = data[numerical_columns].copy()

X = pd.concat([combined_encoded_df, binary_df, numerical_df], axis=1)

y = data[['net_sentiment', 'positivity_ratio']]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Scale numerical features only
X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Train a Random Forest Regressor model
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
model = MultiOutputRegressor(base_model)
model.fit(X_train_scaled, y_train)
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

# FIXME: Decision Tree