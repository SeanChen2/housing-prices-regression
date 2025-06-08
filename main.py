from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import sklearn

# Loads the data from the 1990 California Housing Prices dataset
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")

    # If the data can't be found, create a datasets directory and load housing.tgz into it
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    
    # Extract the data from the tarball file into datasets
    with tarfile.open(tarball_path) as tarball:
        tarball.extractall(path="datasets")

    # Load the CSV into DataFrame
    return pd.read_csv(Path("datasets/housing/housing.csv"))


if __name__ == "__main__":
    housing_data = load_housing_data()
    print(housing_data.head())
    print(housing_data.info())
    print(housing_data["ocean_proximity"].value_counts())
    print(housing_data.describe())


