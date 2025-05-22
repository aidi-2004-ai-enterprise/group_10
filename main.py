import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def split_dataset():
    # Load the penguins dataset
    penguins = sns.load_dataset("penguins")
    print("Penguins dataset loaded successfully!")
    print(penguins.head())
    print(penguins.info())
    
    # Select features and target
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    target = 'species'
    
    # Drop rows with missing values in selected features or target
    penguins = penguins.dropna(subset=features + [target])
    
    # Encode the target variable (species) as numerical labels
    le = LabelEncoder()
    penguins['species_encoded'] = le.fit_transform(penguins[target])
    
    # Split features and target
    X = penguins[features]
    y = penguins['species_encoded']
    
    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nDataset split successfully!")
    print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Testing set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, le

def create_xgboost_model():
    # Create a default XGBoost classifier
    model = XGBClassifier(
        objective='multi:softmax',  # For multiclass classification
        num_class=3,               # Number of classes in penguins dataset (Adelie, Chinstrap, Gentoo)
        eval_metric='mlogloss',    # Multiclass log loss for evaluation
        random_state=42            # For reproducibility
    )
    
    print("Default XGBoost model created successfully!")
    print("Model parameters:", model.get_params())
    
    return model

def fit_model(model, X_train, X_test, y_train, y_test, le):
    # Fit the model on the training data
    model.fit(X_train, y_train)
    print("Model fitted successfully!")
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {accuracy:.4f}")
    
    # Decode predictions to original species names for a sample
    y_pred_labels = le.inverse_transform(y_pred[:5])
    print("Sample predictions (first 5):", y_pred_labels)
    
    return model

if __name__ == "__main__":
    # Execute the workflow
    X_train, X_test, y_train, y_test, le = split_dataset()  
    model = create_xgboost_model()                         
    fitted_model = fit_model(model, X_train, X_test, y_train, y_test, le)  