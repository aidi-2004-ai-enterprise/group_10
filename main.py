import seaborn as sns 
import xgboost as xgb 

def main():
    penguins = sns.load_dataset("penguins")
    print("Penguins dataset loaded successfully!")
    print(penguins.head())
    print(penguins.info())


if __name__ == "__main__":
    main()


# Create a default XGBoost model
model = xgb.XGBClassifier()
print("Default XGBoost model created.")