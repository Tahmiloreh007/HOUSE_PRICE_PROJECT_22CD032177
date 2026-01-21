import os
import joblib
import numpy as np
from flask import Flask, render_template, request

# 1. Initialize the Flask App
app = Flask(__name__)

# 2. MODEL LOADING SECTION (Place it here)
# This gets the directory where app.py actually lives
base_path = os.path.dirname(os.path.abspath(__file__))
# This joins that path with 'model' and your filename
model_path = os.path.join(base_path, 'model', 'house_price_model.pkl')

# Now load using the full path
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model loaded successfully!")
else:
    model = None
    print(f"ERROR: Could not find model at {model_path}")

# 3. ROUTES SECTION
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model not loaded.")
    
    try:
        # Collect features from the HTML form (must match your 6 features)
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Estimated House Price: ${output:,}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)