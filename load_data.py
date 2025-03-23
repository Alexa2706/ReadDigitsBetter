import pickle
import os
import time
import pandas as pd
def load_dataset():
    train_path = "train_cache.pkl"
    test_path = "test_cache.pkl"
    if os.path.exists(train_path):
        assert os.path.exists(test_path) #ako postoji jedan cache, mora postojati i drugi
        print("Loading dataset from cache...")
        start_time = time.time()
        with open(train_path, 'rb') as f:
            train = pickle.load(f) 
        with open(test_path, 'rb') as f:
            test = pickle.load(f)
        print(f"Dataset loaded from cache in {time.time() - start_time:.2f} seconds")
    else:
        print("Processing dataset and creating cache...")
        start_time = time.time()
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
        print(f"Dataset processed in {time.time() - start_time:.2f} seconds")
        with open(train_path, 'wb') as f:
            pickle.dump(train, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test, f)
        print("Dataset cached for future use")
    return train, test
