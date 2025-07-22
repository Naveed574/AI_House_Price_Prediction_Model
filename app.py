from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load once at startup
model = load_model("my_model.h5")
scaler = joblib.load('feature_scaler.pkl')
original_columns = joblib.load('final_columns.pkl')  # 272 feature names
y_scaler = joblib.load('label_scaler.pkl')
template_row = joblib.load('template_row.pkl')  # Already dummy-encoded, not scaled
form_to_model_map = {
    'sqft': 'GrLivArea',
    'bedrooms': 'BedroomAbvGr',
    'bathrooms': 'FullBath',
    'location': 'Neighborhood',
    'property_type': 'HouseStyle',
    'year_built': 'YearBuilt',
    'parking': 'GarageType',
    'basement_finish': 'BsmtFinType1',
    'condition': 'OverallQual',
    'view': 'Exterior1st'
}



def postprocess_prediction(pred_scaled):
    pred_unscaled = y_scaler.inverse_transform(pred_scaled)
    pred_original = np.expm1(pred_unscaled)
    return pred_original

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/form')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse form input
        form_data = request.form.to_dict()
        print("\n==== Raw form data ====\n", form_data)

        # Convert inputs
        for key in form_data:
            try:
                form_data[key] = float(form_data[key])
            except ValueError:
                pass  # Keep as-is

        # 1️⃣ Start with mean-encoded 272-feature row
        user_row = template_row.copy()  # Series

        # 2️⃣ Replace any user inputs
                # Quality mapping for OverallQual
        quality_map = {
            "Poor": 1,
            "Fair": 3,
            "Average": 5,
            "Good": 7,
            "Excellent": 10
        }

        garage_map = {
            "Yes": "Attchd",  # update this to match your training data dummy name
            "No": "Detchd"
        }

        view_map = {
            "good": "VinylSd",       # Replace with your actual Exterior1st values
            "average": "MetalSd",
            "excellent": "BrkFace",
            "poor": "Wd Sdng"
        }

        location_map = {
            "karachi": "NridgHt",    # Replace with real Neighborhood values used in training
            "lahore": "CollgCr",
            "islamabad": "OldTown",
            "peshawar": "Somerst"
        }

        property_type_map = {
            "small_house": "1Story",
            "villa": "2Story",
            "duplex": "1.5Fin",
            "bungalow": "SLvl"
        }

        basement_finish_map = {
            "Unfinished": "Unf",
            "Low": "LwQ",
            "Below Average": "BLQ",
            "Recreation": "Rec",
            "Average": "ALQ",
            "Good": "GLQ",
            "Excellent": "GLQ"
        }


        for form_key, val in form_data.items():
            model_col = form_to_model_map.get(form_key)
            
            if model_col == "BsmtFinType1":
                mapped_val = basement_finish_map.get(str(val).strip().title())
                if mapped_val:
                    dummy_col = f"{model_col}_{mapped_val}"
                    if dummy_col in user_row.index:
                        user_row[dummy_col] = 1
                    else:
                        print(f"⚠️ Dummy column '{dummy_col}' not found in template.")
                else:
                    print(f"⚠️ basement_finish '{val}' not recognized, skipping.")
                continue


            # Special case: map 'property_type' input to real HouseStyle value
            if model_col == "HouseStyle":
                mapped_val = property_type_map.get(str(val).lower())
                if mapped_val:
                    dummy_col = f"{model_col}_{mapped_val}"
                    if dummy_col in user_row.index:
                        user_row[dummy_col] = 1
                    else:
                        print(f"⚠️ Dummy column '{dummy_col}' not found in template.")
                else:
                    print(f"⚠️ Property type '{val}' not recognized, skipping.")
                continue  # done with this feature

            


            # Special case: map 'location' input to real Neighborhood value
            if model_col == "Neighborhood":
                mapped_val = location_map.get(str(val).lower())
                if mapped_val:
                    dummy_col = f"{model_col}_{mapped_val}"
                    if dummy_col in user_row.index:
                        user_row[dummy_col] = 1
                    else:
                        print(f"⚠️ Dummy column '{dummy_col}' not found in template.")
                else:
                    print(f"⚠️ Location '{val}' not recognized, skipping.")
                continue  # done with this feature



            # Special case: map 'view' input to real Exterior1st value
            if model_col == "Exterior1st":
                mapped_val = view_map.get(str(val).lower())
                if mapped_val:
                    dummy_col = f"{model_col}_{mapped_val}"
                    if dummy_col in user_row.index:
                        user_row[dummy_col] = 1
                    else:
                        print(f"⚠️ Dummy column '{dummy_col}' not found in template.")
                else:
                    print(f"⚠️ View '{val}' not recognized, skipping.")
                continue  # done with this feature



            # Skip if no mapping found
            if not model_col:
                print(f"⚠️ Warning: '{form_key}' has no mapping, skipping.")
                continue

            # Special case: map 'condition' (form_key) → 'OverallQual' (model_col)
            if model_col == "OverallQual":
                if val in quality_map:
                    user_row[model_col] = quality_map[val]
                else:
                    print(f"⚠️ Warning: Quality '{val}' not recognized, skipping.")
                continue  # already handled

                    # Special handling for GarageType (parking)
            if model_col == "GarageType":
                val = garage_map.get(val, val)

            # Try dummy column first
            dummy_col = f"{model_col}_{val}"
            if dummy_col in user_row.index:
                user_row[dummy_col] = 1
                continue

            # Try numeric or original column
            if model_col in user_row.index:
                try:
                    user_row[model_col] = float(val)
                except ValueError:
                    user_row[model_col] = val  # in case it's a string feature, not a float
                continue

            print(f"⚠️ Warning: '{form_key}' → '{model_col}' or dummy '{dummy_col}' not found in features, skipping.")



        # 3️⃣ Convert to DataFrame and scale
        user_df = pd.DataFrame([user_row])
        user_scaled = scaler.transform(user_df)

        # 4️⃣ Predict
        pred_scaled = model.predict(user_scaled)
        final_pred = postprocess_prediction(pred_scaled)

        formatted_pred = f"${final_pred[0][0]:,.2f}"
        return render_template('result.html', prediction=formatted_pred)
    
    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><a href='/'>Back</a>"

print("App is starting...")
if __name__ == '__main__':
    app.run(debug=True)
