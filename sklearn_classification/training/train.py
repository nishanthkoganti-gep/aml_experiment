# import modules
import os
import glob
import argparse
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

# aml imports
from azureml.core import Run

# relative imports
from utils import load_data

# user parameters for running training scripts
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder',
                    help='data folder mounting point')
parser.add_argument('--regularization', type=float, dest='reg',
                    default=0.01, help='regularization rate')
args = parser.parse_args()

data_folder = args.data_folder
print('Data folder:', data_folder)

# load train and test set into numpy arrays
X_train = load_data(glob.glob(os.path.join(
    data_folder, '**/train-images-idx3-ubyte.gz'), recursive=True)[0], False) / 255.0
X_test = load_data(glob.glob(os.path.join(
    data_folder, '**/t10k-images-idx3-ubyte.gz'), recursive=True)[0], False) / 255.0
y_train = load_data(glob.glob(os.path.join(
    data_folder, '**/train-labels-idx1-ubyte.gz'), recursive=True)[0], True).reshape(-1)
y_test = load_data(glob.glob(os.path.join(
    data_folder, '**/t10k-labels-idx1-ubyte.gz'), recursive=True)[0], True).reshape(-1)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep = '\n')

# get hold of the current run
run = Run.get_context()

print('Train classifier with regularization: ', args.reg)
clf = LogisticRegression(C=1.0/args.reg, solver="liblinear",
                         multi_class="auto", random_state=42)
clf.fit(X_train, y_train)

print('Predict the test set')
y_hat = clf.predict(X_test)

# calculate accuracy on the prediction
acc = np.average(y_hat == y_test)
print('Accuracy is', acc)

run.log('accuracy', np.float(acc))
run.log('regularization rate', np.float(args.reg))

# save model to file
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=clf, filename='outputs/sklearn_classification_model.pkl')
