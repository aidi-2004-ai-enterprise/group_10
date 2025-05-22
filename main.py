import seaborn as sns  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def split_dataset():
    # Load the penguins dataset
    # Check if seaborn is installed
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn is not installed. Please install it to load the dataset.")
        return
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
    
    print("\nDataset is split successfully!")
    print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Testing set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, le

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le = split_dataset()