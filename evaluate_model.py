import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# --- Step 1: Load the Trained Model ---
# This script assumes 'model.pkl' is in the same directory.
try:
    model = joblib.load('model.pkl')
    print("Model 'model.pkl' loaded successfully.")
except FileNotFoundError:
    print("Error: 'model.pkl' not found. Please run the training script first to create the model file.")
    exit()

# --- Step 2: Load the Test Dataset ---
# This script assumes your new 'test.csv' is in the same directory.
try:
    test_data = pd.read_csv('test.csv')
    print("Test dataset 'test.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'test.csv' not found. Please make sure the test data file is in the same directory.")
    exit()

# --- Step 3: Prepare the Test Data ---
# It's critical that the test data has the exact same columns in the same order as the training data.

# Define the feature columns and the target column.
features = [
    'profile pic', 'nums/length username', 'fullname words', 'nums/length fullname',
    'name==username', 'description length', 'external URL', 'private', '#posts',
    '#followers', '#follows'
]
target = 'fake'

# Check if all required columns exist in the test data
if not all(col in test_data.columns for col in features + [target]):
    print("Error: The test.csv file is missing one or more required columns.")
    print(f"Required columns are: {features + [target]}")
    exit()


X_test = test_data[features]
y_test = test_data[target] # These are the true labels we will compare against.

# Rename columns to match the names used during training.
X_test = X_test.rename(columns={
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


# --- Step 4: Make Predictions on the Test Data ---
print("\nMaking predictions on the new test data...")
predictions = model.predict(X_test)
print("Predictions complete.")

# --- Step 5: Evaluate and Report Performance ---
accuracy = accuracy_score(y_test, predictions)
print(f"\n--- Model Performance on test.csv ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("--------------------------------------\n")

# For a more detailed breakdown (precision, recall, f1-score)
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=['Real (0)', 'Fake (1)']))