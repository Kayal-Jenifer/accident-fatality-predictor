import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, roc_curve, auc)
from imblearn.over_sampling import SMOTE
from folium import Map, Marker, Element
from folium.plugins import MarkerCluster
import os
import joblib
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) # To suppress some joblib/sklearn warnings

# --- 1. Load Data ---
print("--- 1. Loading Data ---")
file_path ="/Users/kayal/Desktop/TOTAL_KSI_6386614326836635957.csv"
df = pd.read_csv(file_path)
    
# --- 2. Initial Exploration ---
print("\n--- 2. Initial Data Exploration ---")
print("\nDataset Info:")
df.info()

print("\nFirst 5 Rows:")
print(df.head())

print("\nSummary Statistics (Numerical):")
# Include 'all' to get stats for object columns too (like counts, unique, top, freq)
print(df.describe(include='all'))

print("\nUnique Value Counts per Column:")
print(df.nunique())

print("\nMissing Values per Column:")
print(df.isnull().sum())

# Check unique values of the target variable BEFORE modification
print("\nUnique values in original 'ACCLASS' column:")
print(df['ACCLASS'].value_counts())

# Correlation Matrix (on original numerical columns)
print("\nCorrelation Matrix (Original Numerical Features):")
numeric_df_orig = df.select_dtypes(include=np.number)
corr_matrix = numeric_df_orig.corr()
print(corr_matrix)


# --- 3. Visualizations ---
# Correlation Matrix & Heatmap
numeric_df = df.select_dtypes(include=np.number)  # Select numerical columns
if not numeric_df.empty:
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.show()
else:
    print("No numerical columns found for correlation heatmap.")

# Histograms for numerical columns
print("\n--- Visualization: Feature Distributions ---")
if not numeric_df.empty:
    numeric_df.hist(figsize=(12, 8), bins=30)
    plt.suptitle("Distribution of Numerical Features")
    plt.show()
else:
    print("No numerical columns found for histograms.")

# Violin Plot for Class-based Feature Distribution
plt.figure(figsize=(10, 6))
sns.violinplot(x='ACCLASS', data=df)
plt.title("Feature Distribution by Class")
plt.show()

# Missing Values Heatmap
print("\n--- Visualization: Missing Values Heatmap ---")
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Folium Map Visualization
print("\n--- Folium Map Visualization ---")
try:
    if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
        map_center = [df["LATITUDE"].mean(), df["LONGITUDE"].mean()]
        mymap = Map(location=map_center, zoom_start=10)
        marker_cluster = MarkerCluster().add_to(mymap)

        for _, row in df.iterrows():
            if pd.notnull(row["LATITUDE"]) and pd.notnull(row["LONGITUDE"]):
                Marker(
                    location=[row["LATITUDE"], row["LONGITUDE"]],
                    popup=f"Lat: {row['LATITUDE']:.4f}, Lon: {row['LONGITUDE']:.4f}"
                ).add_to(marker_cluster)

        mymap.save("visualization.html")
        print("Map saved!")
    else:
        print("Latitude and Longitude columns not found in the dataset.")
except Exception as e:
    print(f"Error creating map visualization: {e}")

# --- 3. Target Variable Preparation ---
print("\n--- 3. Preparing Target Variable (y) ---")

positive_class_label = 'Fatal'

# Create binary target variable y: 1 if Fatal, 0 otherwise
df['TARGET'] = df['ACCLASS'].apply(lambda x: 1 if x == positive_class_label else 0)
y = df['TARGET']

print(f"\nTarget variable 'TARGET' created.")
print(f"Mapping: '{positive_class_label}' -> 1, Others -> 0")
print("Class Distribution (0 = Non-Fatal/Other, 1 = Fatal):")
print(y.value_counts())
print(f"Percentage of positive class (Fatal): {y.mean() * 100:.2f}%")


# --- 4. Feature Selection ---
print("\n--- 4. Selecting Features for Model ---")
# Define the SPECIFIC features requested for the final model
selected_features = ['TIME', 'INVAGE', 'INJURY', 'STREET1', 'STREET2', 'VEHTYPE', 'LIGHT']
print(f"Selected features: {selected_features}")

# Verify all selected features exist in the original DataFrame
missing_selection = [col for col in selected_features if col not in df.columns]
if missing_selection:
    print(f"Error: The following selected features are not in the DataFrame: {missing_selection}")
    exit()

# Create a new DataFrame with only the selected features (and the original index)
df_selected = df[selected_features].copy()
print(f"Created df_selected with shape: {df_selected.shape}")


# --- 5. Data Cleaning/Preprocessing (Selected Features Only) ---
print("\n--- 5. Cleaning and Preprocessing Selected Features ---")

# --- 5a. Handle Missing Values (within df_selected) ---
print("\nHandling missing values in selected features...")
missing_before = df_selected.isnull().sum()
print("Missing values BEFORE handling:")
print(missing_before[missing_before > 0])

# Separate numeric and categorical selected features
selected_numeric_cols = df_selected.select_dtypes(include=np.number).columns.tolist()
selected_categorical_cols = df_selected.select_dtypes(include='object').columns.tolist()

print(f"Selected numeric: {selected_numeric_cols}")
print(f"Selected categorical: {selected_categorical_cols}")

# Fill numeric with mean (or median if preferred)
for col in selected_numeric_cols:
    if df_selected[col].isnull().any():
        mean_val = df_selected[col].mean()
        df_selected[col] = df_selected[col].fillna(mean_val)
        print(f"  Filled NaNs in numeric '{col}' with mean ({mean_val:.2f})")

# Fill categorical with mode
for col in selected_categorical_cols:
    if df_selected[col].isnull().any():
        mode_val = df_selected[col].mode()[0]
        df_selected[col] = df_selected[col].fillna(mode_val)
        print(f"  Filled NaNs in categorical '{col}' with mode ('{mode_val}')")

missing_after = df_selected.isnull().sum().sum()
print(f"\nTotal missing values AFTER handling: {missing_after}")
if missing_after > 0:
    print("Warning: Some missing values might remain.")
    print(df_selected.isnull().sum())


# --- 5b. TIME column check (Assuming integer HHMM format) ---

if 'TIME' in df_selected.columns:
     print(f"\n'TIME' column dtype: {df_selected['TIME'].dtype}")


# --- 5c. Encode Categorical Features & Save Encoders/Unique Values ---
print("\nEncoding selected categorical features and saving artifacts...")
encoder_dir = 'encoders'
os.makedirs(encoder_dir, exist_ok=True)
print(f"Ensured encoder directory exists at: {os.path.abspath(encoder_dir)}")

# Store fitted encoders and unique values
fitted_encoders = {}
unique_values_for_flask = {}

for col in selected_categorical_cols:
    print(f"  Encoding '{col}'...")
    # Ensure column is string type for encoder
    df_selected[col] = df_selected[col].astype(str)

    # Get unique values BEFORE encoding
    unique_vals = sorted(df_selected[col].unique().tolist())
    unique_values_for_flask[col] = unique_vals
    print(f"    Found {len(unique_vals)} unique values for '{col}'.") # Be mindful of high cardinality!

    # Fit LabelEncoder
    le = LabelEncoder()
    df_selected[col] = le.fit_transform(df_selected[col])
    fitted_encoders[col] = le
    print(f"    Encoded '{col}' into numerical representation.")

    # Save the fitted encoder
    if col != 'INVAGE':
        encoder_filepath = os.path.join(encoder_dir, f"{col}_encoder.pkl")
        try:
            joblib.dump(le, encoder_filepath)
            print(f"    -> Saved encoder for '{col}' to {encoder_filepath}")
        except Exception as e:
            print(f"    !!! Error saving encoder for {col}: {e}")
    else:
        print(f"    -> Skipped saving encoder file for '{col}'.")


# Save the unique values dictionary
unique_values_filepath = 'unique_values.pkl'
try:
    joblib.dump(unique_values_for_flask, unique_values_filepath)
    print(f"\nSaved unique values dictionary ({len(unique_values_for_flask)} features) to {unique_values_filepath}")
except Exception as e:
    print(f"\n!!! Error saving unique values dictionary: {e}")


# --- 5d. Prepare Final Feature Matrix X ---
X = df_selected[selected_features] # Ensure order is maintained
print("\nFinal feature matrix X prepared with processed selected features.")
print(f"X shape: {X.shape}")
print("X dtypes:\n", X.dtypes)
print("X head:\n", X.head())


# --- 6. Train/Test Split ---
print("\n--- 6. Splitting Data into Training and Testing Sets ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("\nClass distribution in y_train:")
print(y_train.value_counts(normalize=True))
print("\nClass distribution in y_test:")
print(y_test.value_counts(normalize=True))


# --- 7. Scaling Features ---
print("\n--- 7. Scaling Features using StandardScaler ---")
# Fit scaler ONLY on training data, then transform both train and test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully.")
print(f"X_train_scaled shape: {X_train_scaled.shape}")

# Save the fitted scaler
scaler_filepath = 'scaler.pkl'
try:
    joblib.dump(scaler, scaler_filepath)
    print(f"Saved fitted scaler to {scaler_filepath}")
except Exception as e:
    print(f"!!! Error saving scaler: {e}")


# --- 8. Handle Imbalanced Classes (SMOTE) ---
print("\n--- 8. Handling Imbalance using SMOTE (on training data) ---")
smote = SMOTE(random_state=42)
# Apply SMOTE to the scaled training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"Shape of resampled training data: X={X_train_resampled.shape}, y={y_train_resampled.shape}")
print("\nClass Distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())


# --- 9. Model Training & Hyperparameter Tuning ---
print("\n--- 9. Training and Tuning Models ---")

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000), # Added max_iter
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    # "SVM": SVC(probability=True, random_state=42), # <<< SVM classifier commented out
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
}

# Define parameter grids for GridSearchCV
param_grids = {
    "Logistic Regression": {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['liblinear', 'saga'] # saga handles l1/l2, liblinear good for smaller datasets
    },
    "Decision Tree": {
        'classifier__max_depth': [5, 10, 20, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 3, 5]
    },
    "Random Forest": {
        'classifier__n_estimators': [100, 200], # Reduced for speed, increase if needed
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 3]
    },
    # "SVM": {   #  SVM parameter grid commented out
    #     'classifier__C': [0.1, 1, 10],
    #     'classifier__gamma': ['scale', 'auto', 0.1], # Common gamma values
    #     'classifier__kernel': ['rbf', 'linear'] # RBF and Linear are common choices
    # },
  "Neural Network": {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'classifier__alpha': [0.0001, 0.001, 0.01]
    }

}

best_models = {}
results = {}

# Grid Search loop
for name, model in classifiers.items():
    print(f"\n--- Tuning {name} ---")
    # Create a pipeline for each model FOR GridSearch
    pipeline = Pipeline([
        ('classifier', model)
    ])

    # Use the parameter grid specific to this classifier
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1, verbose=1) # Use accuracy, or f1 maybe better for imbalance

    # Fit GridSearch on the SCALED and RESAMPLED training data
    grid_search.fit(X_train_resampled, y_train_resampled)

    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    print(f"Best CV Score (Accuracy) for {name}: {grid_search.best_score_:.4f}")

    # Store the best estimator (the whole pipeline)
    best_models[name] = grid_search.best_estimator_

    # --- 10. Model Evaluation (on Test Set) ---
    print(f"\n--- Evaluating Best {name} on Test Set ---")
    y_pred = best_models[name].predict(X_test_scaled)
    y_proba = None
    if hasattr(best_models[name], "predict_proba"):
         y_proba = best_models[name].predict_proba(X_test_scaled)[:, 1]
    elif name == "SVM" and hasattr(best_models[name], "decision_function"):
       
        y_scores = best_models[name].decision_function(X_test_scaled)
        
        y_proba = None 
        print(f"Note: Using decision_function scores for potential ROC plot for {name}, storing 'None' for probabilities.")
    else:
        y_proba = None # Model doesn't support probability estimates


    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Confusion Matrix': cm, 'Proba': y_proba}

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
   

# Plot ROC Curves
print("\n--- Plotting ROC Curves ---")
plt.figure(figsize=(10, 8))
if not results:
    print("No model results available to plot ROC curves.")
else:
    for name, res in results.items():
        if res['Proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['Proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        
        else:
            print(f"Cannot plot ROC for {name} as probability/scores are not available.")

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


if not results:
    print("\n--- Recommendation ---")
    print("No models were successfully trained and evaluated.")
    best_model_name = None
else:
    best_model_name = max(results, key=lambda name: results[name]['F1']) # Using F1 score
    print(f"\n--- Recommendation ---")
    print(f"Based on F1 Score on the test set, the recommended model is: {best_model_name}")
    print(f"  - F1 Score: {results[best_model_name]['F1']:.4f}")
    print(f"  - Accuracy: {results[best_model_name]['Accuracy']:.4f}")
    print(f"  - Recall:   {results[best_model_name]['Recall']:.4f}")
    print(f"  - Precision:{results[best_model_name]['Precision']:.4f}")


# --- 11. Save Final Model and Artifacts ---
if best_model_name and best_model_name in best_models:
    print("\n--- 11. Saving Best Model and Necessary Artifacts for Deployment ---")

    # Get the best pipeline object (which only contains the classifier step from GridSearch)
    best_classifier_pipeline = best_models[best_model_name]

    print("Re-creating the best model as a full pipeline (Scaler + Best Classifier)...")
    # Get the best classifier instance from the simplified pipeline
    best_classifier_instance = best_classifier_pipeline.named_steps['classifier']

    # Create the final, deployable pipeline including the scaler fitted earlier
    deployable_pipeline = Pipeline([
        ('scaler', scaler), # Use the scaler fitted on X_train
        ('classifier', best_classifier_instance)
    ])
    print("Deployable pipeline created.")


    # Save the deployable pipeline
    model_filename = 'best_model_pipeline.pkl'
    try:
        joblib.dump(deployable_pipeline, model_filename)
        print(f"Successfully saved the final deployable pipeline to '{model_filename}'")
    except Exception as e:
        print(f"!!! Error saving the final pipeline: {e}")


    # Save the list of feature names (in the correct order the model expects)
    features_filename = 'selected_feature_names.pkl'
    try:
        joblib.dump(selected_features, features_filename)
        print(f"Successfully saved the list of {len(selected_features)} feature names to '{features_filename}'")
    except Exception as e:
        print(f"!!! Error saving feature names: {e}")

    # The scaler and encoders were already saved in steps 7 and 5c.
    print("\n--- Artifacts Saved for Flask Deployment ---")
    print(f"1. Deployable Model Pipeline: {model_filename}")
    print(f"2. Scaler: {scaler_filepath}")
    print(f"3. Feature Names: {features_filename}")
    print(f"4. Label Encoders: In '{encoder_dir}/' directory (e.g., INJURY_encoder.pkl)")
    print(f"5. Unique Categorical Values: {unique_values_filepath}")

elif best_model_name:
     print(f"\n--- Saving Skipped ---")
     print(f"Error: Best model name '{best_model_name}' found, but corresponding model object not in 'best_models'. Check GridSearch results.")
else:
    print("\n--- Saving Skipped ---")
    print("No best model was identified from the evaluation results.")

print("\n--- Model Training and Saving Process Completed ---")












































# Rest of the code is only for testing and it wont work without an additional step
# If in case professor is seeing this you have to make an INVAGE_encoder.pkl from the encoding loop
# Currently the encoding loop doent encode invage as it is being encoded in app.py



'''# --- 12. Test Predictions on Sample Instances (using final pipeline) ---

# Check if necessary components are available before testing
if best_model_name and 'deployable_pipeline' in locals() and 'encoders' in locals() and 'selected_features' in locals():
    print("\n--- 12. Testing Predictions on Sample Instances ---")

    # Define sample instances with raw inputs (like from a form)
    # Make sure the categorical values exist in your actual data/encoders!
    sample_instances_raw = [
        # 1: Young Adult, Minor Injury, Daylight, Auto
        {'hour': 15, 'minute': 30, 'INVAGE': 22, 'INJURY': 'Minor', 'STREET1': 'BATHURST ST', 'STREET2': 'DUNDAS ST W', 'VEHTYPE': 'Automobile, Station Wagon', 'LIGHT': 'Daylight'},
        # 2: Middle Age, Major Injury, Artificial Dark, Truck
        {'hour': 21, 'minute': 0, 'INVAGE': 51, 'INJURY': 'Major', 'STREET1': 'DANFORTH AVE', 'STREET2': 'WOODBINE AVE', 'VEHTYPE': 'Pick Up Truck', 'LIGHT': 'Dark, artificial'},
        # 3: Teenager, Minor Injury, Daylight, Bicycle (Check if 'YONGE ST'/'QUEEN ST W' exist)
        {'hour': 16, 'minute': 10, 'INVAGE': 16, 'INJURY': 'Minor', 'STREET1': 'YONGE ST', 'STREET2': 'QUEEN ST W', 'VEHTYPE': 'Bicycle', 'LIGHT': 'Daylight'},
        # 4: Senior, Major Injury, Dark (Check if 'FINCH AVE W'/'JANE ST' exist)
        {'hour': 2, 'minute': 15, 'INVAGE': 82, 'INJURY': 'Major', 'STREET1': 'FINCH AVE W', 'STREET2': 'JANE ST', 'VEHTYPE': 'Taxi', 'LIGHT': 'Dark'},
    ]

    # Get the final deployable pipeline
    pipeline_to_test = deployable_pipeline

    for i, raw_data in enumerate(sample_instances_raw):
        print(f"\n--- Instance {i+1} ---")
        print(f"Raw Input: {raw_data}")

        instance_processed = [] # To store the processed numerical features
        all_inputs_valid = True # Flag to track validity

        # Process features in the exact order expected by the model
        for feature in selected_features:
            try:
                # --- Handle TIME ---
                if feature == 'TIME':
                    hour = raw_data['hour']
                    minute = raw_data['minute']
                    # Basic validation (could be more robust)
                    if not (0 <= hour <= 23 and 0 <= minute <= 59):
                        print(f"Error: Invalid time in raw data ({hour}:{minute})")
                        all_inputs_valid = False
                        break
                    time_value = float(hour * 100 + minute)
                    instance_processed.append(time_value)
                    # print(f"  Processed TIME -> {time_value}") # Optional debug

                # --- Handle INVAGE ---
                elif feature == 'INVAGE':
                    age = raw_data['INVAGE']
                    # Basic validation
                    if not (isinstance(age, int) and 0 <= age <= 120):
                        print(f"Error: Invalid age value ({age})")
                        all_inputs_valid = False
                        break

                    # Map age to category string
                    age_category = None
                    if 0 <= age <= 4: age_category = "0 to 4"
                    elif 5 <= age <= 9: age_category = "5 to 9"
                    elif 10 <= age <= 14: age_category = "10 to 14"
                    elif 15 <= age <= 19: age_category = "15 to 19"
                    elif 20 <= age <= 24: age_category = "20 to 24"
                    elif 25 <= age <= 29: age_category = "25 to 29"
                    elif 30 <= age <= 34: age_category = "30 to 34"
                    elif 35 <= age <= 39: age_category = "35 to 39"
                    elif 40 <= age <= 44: age_category = "40 to 44"
                    elif 45 <= age <= 49: age_category = "45 to 49"
                    elif 50 <= age <= 54: age_category = "50 to 54"
                    elif 55 <= age <= 59: age_category = "55 to 59"
                    elif 60 <= age <= 64: age_category = "60 to 64"
                    elif 65 <= age <= 69: age_category = "65 to 69"
                    elif 70 <= age <= 74: age_category = "70 to 74"
                    elif 75 <= age <= 79: age_category = "75 to 79"
                    elif 80 <= age <= 84: age_category = "80 to 84"
                    elif 85 <= age <= 89: age_category = "85 to 89"
                    elif 90 <= age <= 94: age_category = "90 to 94"
                    elif age >= 95: age_category = "Over 95"

                    if age_category is None:
                        print(f"Error: Could not map age {age} to category string.")
                        all_inputs_valid = False
                        break

                    # Encode the category string
                    if feature not in fitted_encoders: # Use fitted_encoders from training script
                         print(f"Error: Encoder for {feature} not found in fitted_encoders.")
                         all_inputs_valid = False
                         break
                    encoder = fitted_encoders[feature]
                    if age_category not in encoder.classes_:
                        print(f"Error: Mapped age category '{age_category}' not known to encoder.")
                        all_inputs_valid = False
                        break
                    encoded_value = encoder.transform([age_category])[0]
                    instance_processed.append(encoded_value)
                    # print(f"  Processed INVAGE {age} -> '{age_category}' -> {encoded_value}") # Optional debug

                # --- Handle Other Categorical Features ---
                elif feature in fitted_encoders:
                    value_str = raw_data[feature]
                    encoder = fitted_encoders[feature]
                    if value_str not in encoder.classes_:
                         print(f"Error: Category '{value_str}' for feature '{feature}' not known to encoder.")
                         all_inputs_valid = False
                         break
                    encoded_value = encoder.transform([value_str])[0]
                    instance_processed.append(encoded_value)
                    # print(f"  Processed {feature} '{value_str}' -> {encoded_value}") # Optional debug

                # --- Handle potential numeric features NOT encoded (if any) ---
                # (Your current setup encodes all selected features except TIME)
                else:
                    # This case shouldn't be hit with your current selected_features
                    # If it were a raw numeric feature:
                    # instance_processed.append(float(raw_data[feature]))
                    print(f"Warning: Feature '{feature}' was not handled by specific logic or found in encoders.")
                    # Assume it should be numeric if not handled, but log warning.
                    try:
                         instance_processed.append(float(raw_data[feature]))
                    except ValueError:
                         print(f"Error: Could not convert unhandled feature '{feature}' value '{raw_data[feature]}' to float.")
                         all_inputs_valid = False
                         break

            except KeyError as e:
                print(f"Error: Missing key '{e}' in raw_data for instance {i+1}.")
                all_inputs_valid = False
                break
            except Exception as e_proc:
                print(f"Error processing feature '{feature}' for instance {i+1}: {e_proc}")
                all_inputs_valid = False
                break

        # --- Make Prediction if processing was successful ---
        if all_inputs_valid:
            if len(instance_processed) == len(selected_features):
                # Convert to NumPy array with dtype=float, reshape for pipeline
                input_array = np.array([instance_processed], dtype=float) # Ensure float and 2D

                # Predict using the final deployable pipeline
                try:
                    prediction = pipeline_to_test.predict(input_array)
                    probabilities = pipeline_to_test.predict_proba(input_array)

                    predicted_class = prediction[0] # 0 or 1
                    predicted_label = "Fatal" if predicted_class == 1 else "Non-Fatal"
                    confidence_non_fatal = probabilities[0][0] * 100
                    confidence_fatal = probabilities[0][1] * 100

                    print(f"Processed Input Array: {input_array}")
                    print(f"Prediction: {predicted_label} (Class {predicted_class})")
                    print(f"Probabilities: [Non-Fatal: {confidence_non_fatal:.2f}%, Fatal: {confidence_fatal:.2f}%]")

               
                except ValueError as ve_pred:
                    print(f"Error during prediction: {ve_pred}. Check input array shape/dtype.")
                    print(f"Input array shape: {input_array.shape}, dtype: {input_array.dtype}")
                except Exception as e_pred:
                    print(f"An unexpected error occurred during prediction: {e_pred}")
            else:
                print(f"Error: Processed data length ({len(instance_processed)}) does not match expected feature count ({len(selected_features)}).")
        else:
            print("Skipping prediction due to processing errors.")

else:
    print("\n--- Skipping Sample Instance Prediction ---")
    if not best_model_name:
        print("Reason: No best model was identified.")
    if 'deployable_pipeline' not in locals():
        print("Reason: 'deployable_pipeline' variable not found (likely saving failed).")
    if 'encoders' not in locals():
        print("Reason: 'encoders' dictionary not found (check loading/creation).")
    if 'selected_features' not in locals():
        print("Reason: 'selected_features' list not found.")

# --- End of Model Training Script ---

'''