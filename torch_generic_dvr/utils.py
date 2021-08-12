import pickle
import numpy as np


def load_transfer_function():
    return np.array([0.0, 0.0, 0.0, 0.0,
                     0.0, 0.5, 0.5, 0.0,
                     0.0, 0.5, 0.5, 0.01,
                     0.0, 0.5, 0.5, 0.0,
                     0.5, 0.5, 0.0, 0.0,
                     0.5, 0.5, 0.0, 0.2,
                     0.5, 0.5, 0.0, 0.5,
                     0.5, 0.5, 0.0, 0.2,
                     0.5, 0.5, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0,
                     1.0, 0.0, 1.0, 0.0,
                     1.0, 0.0, 1.0, 0.8]).reshape(12, 4)


def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        return data


def dump_data(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_head_data():
    return load_data("./skewed_head.pickle")
