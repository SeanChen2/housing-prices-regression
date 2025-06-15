import numpy as np
from zlib import crc32

# Split the data into a training set and a test set. The test set is {test_ratio}% of the data
def shuffle_and_split_data(data, test_ratio):
    # Assign random indices in the data to the test and train sets
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    # Return dataframes for the test and train sets that only contain their assigned indices
    return data.iloc[train_indices], data.iloc[test_indices]

# Option 1: Naive split - test set will change when dataset updates (this is bad)
def naive_split(housing_data, test_ratio):
    train_set, test_set = shuffle_and_split_data(housing_data, test_ratio)
    print(len(test_set))


# Checks if an instance should be included in the test set, based on an ID.
# This is a helper hash function for split_data_with_id_hash.
def is_id_in_test_set(id, test_ratio):
    return crc32(np.int64(id)) < test_ratio * 2**32

# Split the data into a training set and a test set, using is_id_in_hash_set
# to determine which set an instance is in
def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# Option 2: Hashing - test set will be retained when the dataset updates (good)
# Could use the index as the ID, but then new data must be added to the end
# of the dataset to prevent index shifts. So I opt for longitude and latitude.
def hash_split(housing_data, test_ratio):
    housing_with_id = housing_data.reset_index()
    housing_with_id["id"] = housing_data["longitude"] * 1000 + housing_data["latitude"]
    train_set, test_set = split_data_with_id_hash(housing_with_id, test_ratio, "id")


