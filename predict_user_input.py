import joblib
import numpy as np

def get_user_input():
    """
    Prompts the user to enter values for each feature and returns them as a list.
    Includes basic validation for numeric inputs.
    """
    print("\nPlease enter the following details for the Instagram account:")
    print("----------------------------------------------------------")

    features = []
    
    # Feature names and their expected types (for validation)
    feature_prompts = {
        'Has Profile Picture (1 for Yes, 0 for No)': int,
        'Ratio of numbers to length of username (e.g., 0.33)': float,
        'Number of words in full name (e.g., 2)': int,
        'Ratio of numbers to length of full name (e.g., 0.0)': float,
        'Is name similar to username (1 for Yes, 0 for No)': int,
        'Length of bio/description (e.g., 150)': int,
        'Has an external URL (1 for Yes, 0 for No)': int,
        'Is the account private (1 for Yes, 0 for No)': int,
        'Number of posts': int,
        'Number of followers': int,
        'Number of following': int
    }

    for prompt, expected_type in feature_prompts.items():
        while True:
            user_input = input(f"- {prompt}: ")
            try:
                # Convert input to the correct type
                value = expected_type(user_input)
                features.append(value)
                break
            except ValueError:
                print(f"  Invalid input. Please enter a valid {'number' if expected_type != float else 'decimal number'}.")
    
    return features

def main():
    """
    Main function to load the model, get user input, and make a prediction.
    """
    # --- Step 1: Load the Trained Model ---
    try:
        model = joblib.load('model.pkl')
        print("Model 'model.pkl' loaded successfully.")
    except FileNotFoundError:
        print("Error: 'model.pkl' not found. Please run the training script first to create the model file.")
        return # Exit the function if model is not found

    # --- Step 2: Get Input from the User ---
    user_features = get_user_input()

    # --- Step 3: Prepare Data and Make Prediction ---
    # Convert the list of features into a NumPy array and reshape it
    # because the model expects a 2D array.
    features_array = np.array(user_features).reshape(1, -1)

    prediction_result = model.predict(features_array)

    # --- Step 4: Display the Result ---
    # The model outputs 1 for 'Fake' and 0 for 'Real'
    final_prediction = 'Fake' if prediction_result[0] == 1 else 'Real'

    print("\n--- Prediction Result ---")
    print(f"The model predicts that this account is: {final_prediction}")
    print("-------------------------\n")


if __name__ == '__main__':
    main()