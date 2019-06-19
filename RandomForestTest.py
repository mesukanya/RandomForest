import numpy as np
import pandas as pd
import pickle


file_name='Randomforest.pkl'

# Loading pickle model to predict data from test_data.csv file
pkl_file = open(file_name, 'rb')
model_pkl = pickle.load(pkl_file)
print("model",model_pkl)

dataSet_testdata = pd.read_csv('test_data.csv')

x1_index = dataSet_testdata.columns.get_loc("Age")
x2_index = dataSet_testdata.columns.get_loc("EstimatedSalary")
y_index = dataSet_testdata.columns.get_loc("Purchased")

print(x1_index)
print(y_index)
x_testdata = dataSet_testdata.iloc[:, [x1_index,x2_index]]
#y_testdata = dataSet_testdata.iloc[:, y_index:(y_index+1)]
y_testdata = dataSet_testdata.iloc[:, [y_index]]
print("x data",x_testdata)
print("y data",y_testdata)

y_pred = model_pkl.predict(x_testdata)

print(y_pred)