import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- Step 1: Load the Dataset ---
# This script assumes 'train.csv' is in the same directory.
try:
    data = pd.read_csv('train.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please make sure the dataset file is in the same directory as this script.")
    exit()

# --- Step 2: Prepare the Data ---
# These are the columns (features) the model will use to make predictions.
# The 'fake' column is our target variable (what we want to predict).
features = [
    'profile pic', 'nums/length username', 'fullname words', 'nums/length fullname',
    'name==username', 'description length', 'external URL', 'private', '#posts',
    '#followers', '#follows'
]
target = 'fake'

X = data[features]
y = data[target]

# Rename columns to be more code-friendly and avoid potential issues.
X = X.rename(columns={
    'profile pic': 'profile_pic',
    'nums/length username': 'nums_length_username',
    'fullname words': 'fullname_words',
    'nums/length fullname': 'nums_length_fullname',
    'name==username': 'name_username_similarity',
    'description length': 'description_length',
    'external URL': 'external_url',
    '#posts': 'posts',
    '#followers': 'followers',
    '#follows': 'following'
})


# --- Step 3: Split Data into Training and Testing Sets ---
# We'll use 80% of the data to train the model and 20% to test its accuracy.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# --- Step 4: Train the Machine Learning Model ---
# We are using a RandomForestClassifier, which is a powerful and popular algorithm
# for classification tasks like this one.
print("Training the model... (This may take a moment)")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available CPU cores
model.fit(X_train, y_train)
print("Model training complete.")

# --- Step 5: Evaluate the Model's Performance ---
# Let's see how well our model performs on the data it hasn't seen before.
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")

# --- Step 6: Save the Trained Model ---
# This saves the trained model to a file named 'model.pkl'.
# Our Flask web app will load this file to make predictions.
joblib.dump(model, 'model.pkl')
print("Model has been saved as 'model.pkl'")