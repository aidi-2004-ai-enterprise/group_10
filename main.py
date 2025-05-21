import seaborn as sns  

def main():
    penguins = sns.load_dataset("penguins")
    print("Penguins dataset loaded successfully!")
    print(penguins.head())
    print(penguins.info())


if __name__ == "__main__":
    main()
