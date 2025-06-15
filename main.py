from pathlib import Path
import pandas as pd
from pandas.plotting import scatter_matrix
import tarfile
import urllib.request
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split

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

    # Load the extracted CSV into a DataFrame object
    return pd.read_csv(Path("datasets/housing/housing.csv"))


# Describe the dataset and generate histograms for each numerical attribute
def describe_data(housing_data):
    print(housing_data.head())
    print(housing_data.info())
    print(housing_data["ocean_proximity"].value_counts())
    print(housing_data.describe())
    housing_data.hist(bins=50, figsize=(12, 8))
    plt.show()


def explore_data(housing_data):
    # Show where the data is most concentrated
    housing_data.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
    plt.show()

    # s means what the size of each dot represents
    # c means what the colour represents, and cmap="jet" means the colour goes from blue to red
    housing_data.plot(kind="scatter", x="longitude", y="latitude", grid=True, 
                      s=housing_data["population"] / 100, label="population",
                      c="median_house_value", cmap="jet", colorbar=True,
                      legend=True, sharex=False, figsize=(10, 7))
    plt.show()

    # Create more useful attribute combinations
    housing_data["rooms_per_house"] = housing_data["total_rooms"] / housing_data["households"]
    housing_data["bedrooms_ratio"] = housing_data["total_bedrooms"] / housing_data["total_rooms"]
    housing_data["people_per_house"] = housing_data["population"] / housing_data["households"]

    # Compute standard correlation coefficient (Pearson's r) between every attribute pair:
    # Closer to 1 means +ve correlation, closer to -1 means -ve correlation, closer to 0 means no linear correlation
    # More extreme standard correlation coefficient does NOT mean higher slope. Just means a more perfect straight line going up/down.
    corr_matrix = housing_data.corr(numeric_only=True)  # Make sure non-numeric data (e.g. ocean_proximity) isn't included
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # Do the same thing with a scatter matrix
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing_data[attributes], figsize=(12, 8))
    plt.show()


if __name__ == "__main__":
    housing_data = load_housing_data()
    # describe_data(housing_data)

    # Since median_income is an important attribute, split it into strata (bins; categories) temporarily
    housing_data["income_cat"] = pd.cut(housing_data["median_income"], 
                                        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                        labels=[1, 2, 3, 4, 5])
    
    # Split the data into 80% training set and 20% test set.
    # Stratify to make sure each set nicely represents each income category (especially test set)
    strat_train_set, strat_test_set = train_test_split(housing_data, test_size=0.2, stratify=housing_data["income_cat"], random_state=42)
    # print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

    # Don't need income_cat anymore, since already stratified in the split
    strat_train_set.drop("income_cat", axis=1, inplace=True)    # axis=1 means drop the COLUMN, axis=0 means drop the INDEX
    strat_test_set.drop("income_cat", axis=1, inplace=True)
    
    housing_data = strat_train_set.copy()   # Don't want to explore any of the test set
    explore_data(housing_data)


