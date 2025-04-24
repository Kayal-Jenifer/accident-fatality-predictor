import pickle
import numpy as np
import os
import joblib
from flask import (Flask, request, render_template,
                   redirect, url_for, flash)
from sklearn.exceptions import NotFittedError
import traceback 
import re

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'c7e6b95c784a3e983baec0a81134ff6e') # Use env var or default

#  Global Variables 
model = None
feature_names = []
encoders = {}
unique_values = {}
loading_error = None # Store loading errors to display

#  Load Model, Scaler , Feature Names, Encoders, Unique Values 
try:
    # paths relative to the app.py file location
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Used abspath for robustness
    model_path = os.path.join(base_dir, 'best_model_pipeline.pkl')
    features_path = os.path.join(base_dir, 'selected_feature_names.pkl')
    encoder_dir = os.path.join(base_dir, 'encoders') # Directory for encoders
    unique_values_path = os.path.join(base_dir, 'unique_values.pkl') # Path for unique values

    print("--- Starting Artifact Loading ---")
    print(f"Base directory: {base_dir}")
    print(f"Looking for model at: {model_path}")
    print(f"Looking for features at: {features_path}")
    print(f"Looking for encoders in: {encoder_dir}")
    print(f"Looking for unique values at: {unique_values_path}")


    # --- Load the trained model (assuming it's a pipeline) ---
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
            print(f"Model pipeline loaded successfully from {model_path}.")
            if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
                 print("Warning: Loaded object might not be a scikit-learn compatible model/pipeline.")
            if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
                 print("Scaler found within the loaded model pipeline.")
            else:
                 print("Warning: Scaler step not explicitly found in pipeline 'named_steps'. Ensure scaling is handled if loaded separately.")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    #  Load feature names 
    if os.path.exists(features_path):
        with open(features_path, 'rb') as f:
            feature_names = joblib.load(f)
            print(f"Feature names loaded successfully from {features_path}: {feature_names}")
        if not isinstance(feature_names, list):
            raise TypeError("selected_feature_names.pkl did not contain a list.")
        if len(feature_names) != 7:
            print(f"Warning: Expected 7 feature names based on training context, but loaded {len(feature_names)}.")
    else:
         raise FileNotFoundError(f"Feature names file not found at {features_path}")

    #  Load the encoders 
    if os.path.isdir(encoder_dir):
        print(f"Loading encoders from directory: {encoder_dir}")
        encoders = {}
        for filename in os.listdir(encoder_dir):
            if filename.endswith("_encoder.pkl"):
                # Extract feature name (e.g., "INJURY_encoder.pkl" -> "INJURY")
                feature__name__from_file = filename.replace("_encoder.pkl", "")
                filepath = os.path.join(encoder_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        encoder = joblib.load(f)
                        # Basic check if it looks like an encoder
                        if hasattr(encoder, 'classes_') and hasattr(encoder, 'transform'):
                             encoders[feature__name__from_file] = encoder
                             print(f" -> Loaded encoder for '{feature__name__from_file}'")
                        else:
                             print(f"Warning: File {filename} loaded but might not be a valid LabelEncoder.")
                except Exception as enc_err:
                    print(f"Error loading encoder file {filename}: {enc_err}")
        print(f"Finished loading encoders. Found {len(encoders)} encoders.")
        # Verify encoders loaded match categorical features expected
        expected_categoricals = ['INJURY', 'STREET1', 'STREET2', 'VEHTYPE', 'LIGHT'] # Based on training script context
        loaded_encoder_keys = list(encoders.keys())
        print(f"Loaded encoder keys: {loaded_encoder_keys}")
        missing_encoders = [cat_feat for cat_feat in expected_categoricals if cat_feat not in loaded_encoder_keys]
        if missing_encoders:
             print(f"Warning: Expected encoders for {missing_encoders} but they were not loaded from '{encoder_dir}'.")

    else:
        print(f"Warning: Encoder directory '{encoder_dir}' not found. Categorical features cannot be processed.")
        

    # --- Load unique values for dropdowns ---
    if os.path.exists(unique_values_path):
        with open(unique_values_path, 'rb') as f:
            unique_values = joblib.load(f)
            print(f"Loaded unique values for dropdowns from {unique_values_path}: {list(unique_values.keys())}")
            missing_unique_vals = [cat_feat for cat_feat in expected_categoricals if cat_feat not in unique_values]
            if missing_unique_vals:
                 print(f"Warning: Expected unique values for {missing_unique_vals} but they were not loaded from '{unique_values_path}'. Dropdowns might be incomplete.")
    else:
        print(f"Warning: Unique values file '{unique_values_path}' not found. Dropdowns may not populate correctly.")

    print("--- Artifact Loading Finished ---")

# ---  Loading Errors ---
except FileNotFoundError as e:
    print(f"Fatal Error: Required file not found during loading: {e}")
    traceback.print_exc()
    loading_error = f"Server Configuration Error: Missing required file ({e}). Please check file paths and ensure artifacts are present. Contact the administrator if needed."
except (pickle.UnpicklingError, joblib.externals.loky.backend.exceptions.BadPickleGetState, TypeError, ValueError, AttributeError) as e:
    print(f"Fatal Error: An error occurred during artifact unpickling or validation: {e}")
    traceback.print_exc()
    loading_error = "Server Configuration Error: Could not load or validate model/data files (possible corruption or version mismatch). Please contact the administrator."
except Exception as e:
    print(f"Fatal Error: An unexpected error occurred during loading: {e}")
    traceback.print_exc()
    loading_error = "Server Configuration Error: An unexpected issue occurred during setup. Please contact the administrator."


# --- Define Routes ---

@app.route('/')
def index():
    """Renders the home/introduction page."""
    if loading_error:
        flash(loading_error, "error")
        return render_template('error.html', error_message=loading_error)
    if model is None or not feature_names:
        error_msg = "Model or feature names failed to load properly. Check server logs and ensure artifact files are correct."
        flash(error_msg, "error")
        return render_template('error.html', error_message=error_msg)

    # Example Model Evaluations 
    model_evaluations = {
        'Logistic Regression': {'Accuracy': 0.62, 'Precision': 0.91, 'Recall': 0.62, 'F1': 0.74}, # Example F1
        'Decision Tree': {'Accuracy': 0.85, 'Precision': 0.93, 'Recall': 0.89, 'F1': 0.91}, # Example F1
        'Random Forest': {'Accuracy': 0.91, 'Precision': 0.93, 'Recall': 0.96, 'F1': 0.94}, # Example Best F1
        'Neural Network': {'Accuracy': 0.77, 'Precision': 0.92, 'Recall': 0.80, 'F1': 0.86}, # Example F1
    }
    # Determine best based on training script output 
    best_model_pipeline_name = max(model_evaluations, key=lambda k: model_evaluations[k]['F1']) # Find best by F1

    return render_template('index.html',
                           evaluations=model_evaluations,
                           best_model_pipeline=best_model_pipeline_name)

@app.route('/predict_form')
def predict_form():
    """Renders the prediction input form page."""
    if loading_error:
        flash(loading_error, "error")
        return render_template('error.html', error_message=loading_error)
    if model is None or not feature_names:
        flash("Cannot proceed: Model/Features not loaded correctly. Check server logs.", "error")
        return redirect(url_for('index'))

    # Pass feature names AND unique values
    return render_template('predict_form.html',
                           feature_names=feature_names,
                           unique_values=unique_values) # Pass the loaded unique values dict

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if loading_error:
        flash(loading_error, "error")
        return redirect(url_for('index'))
    if model is None or not feature_names:
        flash("Cannot predict: Model/Features not loaded correctly.", "error")
        return redirect(url_for('predict_form'))
    # Check if encoders are needed and loaded
    categorical_features_expected = [f for f in feature_names if f in ['INJURY', 'STREET1', 'STREET2', 'VEHTYPE', 'LIGHT']]
    if categorical_features_expected and not encoders:
         flash("Cannot predict: Encoders required for categorical features but were not loaded.", "error")
         return redirect(url_for('predict_form'))

    if request.method == 'POST':
        try:
            input_data = []
            form_data_display = {} # Store user-friendly input for display

            # --- Get all form data first for easier access ---
            form_values = request.form.to_dict()
            print(f"Received form data: {form_values}")

            # --- Process input based on expected feature_names order ---
            for feature in feature_names:

                #  SPECIAL HANDLING FOR TIME
                if feature == 'TIME':
                    hour_str = form_values.get('hour')
                    minute_str = form_values.get('minute')
                    # Store display value (handle potential None)
                    form_data_display[feature] = f"{hour_str or '?'}:{minute_str or '?'}" # Display as H:M

                    # Validation
                    if hour_str is None or minute_str is None or not hour_str.isdigit() or not minute_str.isdigit():
                        flash("Error: Both Hour and Minute must be provided as numbers for TIME.", "error")
                        return redirect(url_for('predict_form'))

                    hour = int(hour_str)
                    minute = int(minute_str)

                    if not (0 <= hour <= 23 and 0 <= minute <= 59):
                        flash("Error: Invalid Hour (0-23) or Minute (0-59) entered.", "error")
                        return redirect(url_for('predict_form'))

                    # Combine into the HHMM integer format
                    time_value = hour * 100 + minute
                    # Append the combined value AS FLOAT (scaler expects float)
                    input_data.append(float(time_value))
                    print(f"  Processed TIME: {hour_str}:{minute_str} -> {float(time_value)}") # Debug

                # --- PROCESS ALL OTHER FEATURES ---
                else:
                    value_str = form_values.get(feature)
                     # Store raw input for display (handle potential None)
                    form_data_display[feature] = value_str if value_str is not None else '?'

                    if value_str is None or value_str.strip() == '':
                        # Allow empty string for features that might legitimately be empty if handled later
                        # For now, enforce all fields are filled
                        flash(f"Error: Missing value for feature '{feature}'. Please fill all fields.", "error")
                        return redirect(url_for('predict_form'))

                    # --- Transformation Logic for cat features ---
                    try:
                        # Categorical Feature: Use loaded encoder
                        if feature in encoders:
                            # Check if encoder actually loaded for this feature
                            if feature not in encoders:
                                 flash(f"Configuration Error: Encoder for '{feature}' was expected but not loaded. Cannot process input.", "error")
                                 return redirect(url_for('predict_form'))

                            encoder = encoders[feature]
                            # Check if the value is known to the encoder
                            if value_str not in encoder.classes_:
                                 known_classes_preview = ", ".join(list(encoder.classes_)[:10]) + ('...' if len(encoder.classes_) > 10 else '')
                                 flash(f"Error: Invalid category '{value_str}' provided for '{feature}'. Known options start with: {known_classes_preview}", "error")
                                 return redirect(url_for('predict_form'))
                            encoded_value = encoder.transform([value_str])[0]
                            input_data.append(encoded_value)
                            print(f"  Processed Categorical '{feature}': '{value_str}' -> {encoded_value} (type: {type(encoded_value).__name__})") # Debug

                        # INVAGE Feature 
                        elif feature == 'INVAGE':
                            value_str = value_str.strip() # Remove whitespace

                            # Allow EITHER a range string OR a single age number
                            processed_value = None 
                            print_msg = "" # For debug printing

                            match_range = re.match(r"^\s*(\d+)\s*(?:to|-|–|\s+)\s*(\d+)\s*$", value_str)

                            if match_range:
                                try:
                                    lower_age = int(match_range.group(1))
                                    upper_age = int(match_range.group(2))

                                    # Add reasonable validation for range values
                                    if not (0 <= lower_age <= 120 and 0 <= upper_age <= 120):
                                        flash(f"Error: Invalid age values in range for ('{feature}'). Ages must be between 0 and 120. Got: {lower_age}, {upper_age}", "error")
                                        return redirect(url_for('predict_form'))

                                    if lower_age > upper_age:
                                        flash(f"Error: Invalid age range for ('{feature}'). Lower age ({lower_age}) cannot be greater than upper age ({upper_age}).", "error")
                                        return redirect(url_for('predict_form'))

                                    # Calculate the midpoint
                                    processed_value = (lower_age + upper_age) / 2.0
                                    print_msg = f"  Processed INVAGE (range): {value_str} -> {processed_value}"

                                except ValueError:
                                    flash(f"Error: Could not parse numbers in age range ('{feature}'). Input: '{value_str}'", "error")
                                    return redirect(url_for('predict_form'))
                                except Exception as e: # Catch unexpected errors during range processing
                                    flash(f"An unexpected error occurred processing age range '{value_str}': {e}", "error")
                                    return redirect(url_for('predict_form'))

                            # If not a range, check if it's a single valid age number
                            elif value_str.isdigit():
                                try:
                                    value = int(value_str)
                                    # Add reasonable validation
                                    if not 0 <= value <= 120: # Example range
                                        flash(f"Error: Invalid value for Age ('{feature}'). Please enter a realistic age (e.g., 0-120). Got: '{value_str}'", "error")
                                        return redirect(url_for('predict_form'))

                                    # Use the single age directly as a float
                                    processed_value = float(value)
                                    print_msg = f"  Processed INVAGE (single): {value_str} -> {processed_value}"

                                except ValueError:
                                    flash(f"Error: Could not parse single age number ('{feature}'). Input: '{value_str}'", "error")
                                    return redirect(url_for('predict_form'))
                                except Exception as e: # Catch unexpected errors during single age processing
                                    flash(f"An unexpected error occurred processing single age '{value_str}': {e}", "error")
                                    return redirect(url_for('predict_form'))

                            else:
                                
                                flash(f"Error: Invalid format for Age ('{feature}'). Expected 'min to max' (e.g., '55 to 59') or a single whole number. Got: '{value_str}'", "error")

                                return redirect(url_for('predict_form'))

                            # Append the calculated float value
                            if processed_value is not None:
                                input_data.append(processed_value) # Append the float
                                print(print_msg) # Debug
                            else:
                                # Fallback error
                                flash(f"Error: Failed to process value for Age ('{feature}'). Input: '{value_str}'", "error")
                                return redirect(url_for('predict_form'))
                
                        else:
                            # Attempt to convert to float, catch error if not possible
                            try:
                                value = float(value_str)
                                input_data.append(value) # Append float
                                print(f"  Processed Numeric '{feature}': {value_str} -> {value}") # Debug
                            except ValueError:
                                flash(f"Error: Invalid numeric value '{value_str}' for feature '{feature}'. Please enter a valid number.", "error")
                                return redirect(url_for('predict_form'))


                    # Catch errors during the value processing for this specific feature
                    except ValueError as ve: # Catch specific numeric conversion errors
                        flash(f"Error: Invalid value or format for '{feature}': '{value_str}'. {str(ve)}", "error")
                        return redirect(url_for('predict_form'))
                    except KeyError:
                        flash(f"Error: Configuration problem. Encoder not found for feature '{feature}'. Check server setup.", "error")
                        return redirect(url_for('predict_form'))
                    except Exception as ex_proc: # Catch any other unexpected error during processing
                        flash(f"An unexpected error occurred processing value '{value_str}' for feature '{feature}': {ex_proc}", "error")
                        print(f"Error processing feature '{feature}':")
                        traceback.print_exc() # Log details
                        return redirect(url_for('predict_form'))

            # --- Post-Processing Checks and Prediction ---

            # Ensure we got the correct number of processed inputs
            if len(input_data) != len(feature_names):
                flash(f"Error: Internal processing mismatch. Expected {len(feature_names)} features, processed {len(input_data)}. Check server logs and feature processing logic.", "error")
                print(f"Debug: Final input_data before numpy conversion: {input_data}")
                return redirect(url_for('predict_form'))

           
            input_array = np.array(input_data, dtype=float).reshape(1, -1)
            print(f"Final input array (before model pipeline): {input_array}") # Debug print
            print(f"Input array dtype: {input_array.dtype}") # Should be float64

           
            prediction = model.predict(input_array)
            prediction_proba = model.predict_proba(input_array)

            print(f"Raw Prediction: {prediction}")
            print(f"Raw Probabilities: {prediction_proba}")

            
            predicted_class_index = prediction[0]
            if predicted_class_index == 1:
                result_text = "Fatal / High Severity Prediction"
                result_class = "fatal" # For CSS styling
            else: # predicted_class_index == 0
                result_text = "Non-Fatal / Lower Severity Prediction"
                result_class = "non-fatal" # For CSS styling

            confidence = prediction_proba[0][predicted_class_index] * 100

            return render_template('result.html',
                                   prediction_text=result_text,
                                   prediction_class=result_class,
                                   confidence=f"{confidence:.2f}%",
                                   user_input=form_data_display) # Show user's original input

        except NotFittedError:
             flash("Error: Model components (like scaler) seem not fitted. This might indicate an issue with the saved pipeline. Please retrain and save the pipeline.", "error")
             print("Error: NotFittedError encountered during prediction.")
             traceback.print_exc()
             return redirect(url_for('predict_form'))
        except ValueError as ve_pred:
             flash(f"Error during prediction processing: {ve_pred}. Check if input data types or number of features match model expectations.", "error")
             print(f"ValueError during prediction: {ve_pred}")
             print(f"Input array shape: {input_array.shape}, dtype: {input_array.dtype}") # Log details
             traceback.print_exc()
             return redirect(url_for('predict_form'))
        except Exception as e_pred: # Catch any other unexpected error during prediction
            print(f"Fatal Error during prediction execution: {e_pred}") # Log the full error
            traceback.print_exc() # Print detailed traceback to console
            flash(f"An unexpected error occurred while making the prediction. Please check server logs or contact the administrator.", "error")
            return redirect(url_for('predict_form'))

    # Redirect back to form if GET request reaches /predict
    return redirect(url_for('predict_form'))

# --- Error Handler for general 404 Not Found ---
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_message="404 - Page Not Found"), 404

# --- Error Handler for general 500 Internal Server Error ---
@app.errorhandler(500)
def internal_server_error(e):
    
    print(f"Internal Server Error triggered: {e}")
    traceback.print_exc() # Log the error stack trace
    return render_template('error.html', error_message="500 - Internal Server Error. An unexpected issue occurred on the server. Please contact the administrator."), 500


# --- Run the App ---
if __name__ == '__main__':
   
    app.run(debug=False, host='127.0.0.1', port=5000)