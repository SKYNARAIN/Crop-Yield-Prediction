import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = 'FinalDS.csv'  # Replace with your file's path
data = pd.read_csv(file_path)

# Preprocess the data
data = data[data['RAINFALL_YEARLY'] != -1]
data.rename(columns={'NITROGEN_CONSUMOTION': 'NITROGEN_CONSUMPTION'}, inplace=True)

# Select features and target variable
features = ['Dist Name', 'Crop', 'AREA', 'NITROGEN_CONSUMPTION', 'PHOSPHATE_CONSUMPTION', 
            'POTASH_CONSUMPTION', 'RAINFALL_YEARLY']
target = 'PRODUCTION'
X = data[features]
y = data[target]

# Preprocessing: One-hot encode categorical columns
categorical_features = ['Dist Name', 'Crop']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# Define the pipeline with a Random Forest model
model = Pipeline(steps=[('preprocessor', preprocessor),
                         ('regressor', RandomForestRegressor(random_state=42))])

# Train the model
model.fit(X, y)

# Initialize Flask app
app = Flask(__name__)

# Home route to show the input form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and predict production
@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    dist_name = request.form['dist_name']
    crop_name = request.form['crop_name']
    area = float(request.form['area'])
    nitrogen = float(request.form['nitrogen'])
    phosphate = float(request.form['phosphate'])
    potash = float(request.form['potash'])
    rainfall = float(request.form['rainfall'])

    # Create a DataFrame for the user input
    input_data = pd.DataFrame({
        'Dist Name': [dist_name],
        'Crop': [crop_name],
        'AREA': [area],
        'NITROGEN_CONSUMPTION': [nitrogen],
        'PHOSPHATE_CONSUMPTION': [phosphate],
        'POTASH_CONSUMPTION': [potash],
        'RAINFALL_YEARLY': [rainfall]
    })
    
    # Predict production
    prediction = model.predict(input_data)[0]

    # Calculate yield
    yield_value = (prediction * 1000000) / (area * 1000)

    return render_template('result.html', production=prediction, yield_value=yield_value)

if __name__ == '__main__':
    app.run(debug=True)