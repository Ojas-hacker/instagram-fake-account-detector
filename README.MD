# AI-Powered Instagram Account Authenticity Analyzer

This project is an AI-powered tool for automated social media forensics and analysis, specifically designed to predict whether an Instagram account is **real** or **fake**. It uses machine learning to analyze Instagram profile features and provides predictions via a web interface and command-line utilities.

## Features

- **Web App**: Enter an Instagram username and get a real/fake prediction instantly.
- **Command-Line Prediction**: Input features manually for prediction without scraping.
- **Model Training & Evaluation**: Scripts to train and evaluate the machine learning model.
- **Data Logging**: All predictions are saved to a CSV file for further analysis.

## Project Structure

```
.
├── app.py                  # Flask web application
├── predict_user_input.py   # CLI tool for manual feature input and prediction
├── train_model.py          # Script to train the ML model
├── evaluate_model.py       # Script to evaluate the model on test data
├── model.pkl               # Trained machine learning model (generated)
├── train.csv               # Training dataset
├── test.csv                # Test dataset
├── predictions.csv         # Log of predictions made via the web app
├── templates/
│   └── index.html          # HTML template for the web interface
└── demo_pics/              # Demo images (not used in code)
```

## Setup Instructions

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/instagram-fake-account-detector.git
cd instagram-fake-account-detector
```

### 2. Install Dependencies

Install Python packages (preferably in a virtual environment):

```sh
pip install -r requirements.txt
```

**Required packages:**
- Flask
- instaloader
- numpy
- pandas
- scikit-learn
- joblib

### 3. Prepare Datasets

Ensure `train.csv` and `test.csv` are present in the project root. These files contain the features and labels for training and evaluating the model.

### 4. Train the Model

Run the training script to generate `model.pkl`:

```sh
python train_model.py
```

### 5. Evaluate the Model (Optional)

Check model performance on the test set:

```sh
python evaluate_model.py
```

### 6. Run the Web Application

Start the Flask server:

```sh
python app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 7. Command-Line Prediction

To predict using manual feature input (without scraping):

```sh
python predict_user_input.py
```

Follow the prompts to enter feature values.

## How It Works

- **Web App**: Enter an Instagram username. The app scrapes public profile data using `instaloader`, extracts relevant features, and predicts authenticity using the trained model.
- **CLI Tool**: Manually input feature values for prediction.
- **Model**: A Random Forest classifier trained on labeled Instagram account data.

## File Descriptions

- [`app.py`](app.py): Main Flask app. Handles user input, scraping, prediction, and result display.
- [`predict_user_input.py`](predict_user_input.py): CLI tool for manual predictions.
- [`train_model.py`](train_model.py): Trains the Random Forest model and saves it as `model.pkl`.
- [`evaluate_model.py`](evaluate_model.py): Evaluates the trained model on `test.csv`.
- [`model.pkl`](model.pkl): Trained model file (generated after training).
- [`predictions.csv`](predictions.csv): Stores all predictions made via the web app.
- [`templates/index.html`](templates/index.html): Web UI template.

## Notes

- **Instagram Scraping**: The web app uses `instaloader` to fetch public profile data. Private or restricted accounts may not be accessible.
- **Data Privacy**: Only public data is used. No login or sensitive information is required.
- **Limitations**: The model's accuracy depends on the quality and diversity of the training data.

## License

This project is licensed under the MIT License.

---

**Contributions and