import instaloader
import numpy as np
import joblib
import csv
import os
from flask import Flask, render_template, request, flash, redirect, url_for

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key_for_flask'

# --- CSV File Configuration ---
CSV_FILE = 'predictions.csv'
CSV_HEADER = [
    'username', 'profile pic', 'nums/length username', 'fullname words', 
    'nums/length fullname', 'name==username', 'description length', 
    'external URL', 'private', '#posts', '#followers', '#follows', 'prediction'
]


# --- Load the Trained Model ---
try:
    model = joblib.load('model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'model.pkl' not found. The application will not be able to make predictions.")
    model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None


def scrape_and_process_profile(username):
    """
    Scrapes an Instagram profile and processes the data into features for the ML model.
    """
    L = instaloader.Instaloader()
    try:
        print(f"Fetching data for profile: {username}...")
        profile = instaloader.Profile.from_username(L.context, username)

        # --- Feature Extraction ---
        digit_count_username = sum(c.isdigit() for c in profile.username)
        username_length = len(profile.username)
        nums_len_username = digit_count_username / username_length if username_length > 0 else 0
        full_name_words = len(profile.full_name.split()) if profile.full_name else 0
        digit_count_fullname = sum(c.isdigit() for c in profile.full_name)
        fullname_length = len(profile.full_name)
        nums_len_fullname = digit_count_fullname / fullname_length if fullname_length > 0 else 0

        # --- Feature List Creation (in the correct order for the model) ---
        features = [
            1 if profile.profile_pic_url else 0,
            round(nums_len_username, 4),
            full_name_words,
            round(nums_len_fullname, 4),
            1 if profile.username.lower() == profile.full_name.lower() else 0,
            len(profile.biography),
            1 if profile.external_url else 0,
            1 if profile.is_private else 0,
            profile.mediacount,
            profile.followers,
            profile.followees
        ]
        
        print(f"Successfully extracted features for {username}.")
        return features, None

    except instaloader.exceptions.ProfileNotFoundError:
        return None, f"Error: Profile '{username}' not found."
    except instaloader.exceptions.LoginRequiredError:
        return None, "Error: This profile is private or requires login. Cannot analyze."
    except Exception as e:
        return None, f"An unexpected error occurred during scraping: {e}"

def save_to_csv(data_row):
    """Appends a new row of data to the predictions.csv file."""
    # Check if file exists to decide whether to write the header
    file_exists = os.path.isfile(CSV_FILE)
    
    try:
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header only if the file is new
            if not file_exists or os.path.getsize(CSV_FILE) == 0:
                writer.writerow(CSV_HEADER)
            writer.writerow(data_row)
        print(f"Successfully saved data to {CSV_FILE}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")


@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles both displaying the form and processing the prediction."""
    if request.method == 'POST':
        if model is None:
            flash("Machine Learning model is not loaded. Please check the server logs.")
            return render_template('index.html')

        username = request.form.get('username')
        if not username:
            flash('Username is required!')
            return redirect(url_for('index'))

        features, error = scrape_and_process_profile(username)

        if error:
            flash(error)
            return render_template('index.html')

        try:
            # Make prediction
            final_features = np.array(features).reshape(1, -1)
            result = model.predict(final_features)
            prediction = 'Fake' if result[0] == 1 else 'Real'
            
            # Prepare data row for CSV
            data_to_save = [username] + features + [prediction]
            
            # Save the result to CSV
            save_to_csv(data_to_save)
            
            # Render the page again with the prediction result
            return render_template('index.html', prediction=prediction, username=username)
        
        except Exception as e:
            flash(f"An error occurred during prediction: {e}")
            return render_template('index.html')

    # For a GET request, just show the page
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
