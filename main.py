from xgboost import XGBClassifier

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

if __name__ == "__main__":
    model = create_xgboost_model()