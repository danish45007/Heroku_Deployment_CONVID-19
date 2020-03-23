import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == "__main__":
    # Read the dataset
    data = pd.read_csv("C:\\Users\\DANISH\\Documents\\convid-19_toy.csv")
    # Train-Test split
    train, test = data_split(data, 0.2)
    X_train = train[['fever', 'bodypain', 'age', 'runnynoise', 'diffBreath']].to_numpy()
    X_test = test[['fever', 'bodypain', 'age', 'runnynoise', 'diffBreath']].to_numpy()
    Y_train = train[['infectionProb']].to_numpy().reshape(2053, )
    Y_test = test[['infectionProb']].to_numpy().reshape(513, )
    # Model Creation
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    # Open a file where you want to store the data
    file = open('model.pkl', 'wb')

    # Dump information into a file
    pickle.dump(clf,file)

    # Close the file
    file.close()

    # Random input Feature
    input_features = [102, 1, 22, -1, 1]
    infProb = clf.predict_proba([input_features])[0][1]
    print(infProb)




