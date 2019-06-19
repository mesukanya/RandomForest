import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
print(x_train)
x_test = sc.transform(x_test)
print("y test",y_test)

#Fitting Random forest classifier
classifier = RandomForestClassifier(n_estimators= 10, criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

#Predecting the test set result
ypred = classifier.predict(x_test)
print(ypred)

#Making the confusion matrix
cm = confusion_matrix(y_test, ypred)
print(cm)

#Visualizing the trainig set result
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start= x_set[:,0].min()-1, stop= x_set[:,0].max()+1, step= 0.01),
np.arange(start= x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step= 0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x2.max())
plt.ylim(x1.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set== j, 0], x_set[y_set== j, 1],
                c = ListedColormap(('red','green'))(i), label=j)
plt.title('Random Forest Classification(Training set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#importing csv
dataset=pd.read_csv('Social_Network_Ads.csv')

#heading categorical data
gender=pd.get_dummies(dataset['Gender'])
dataset=dataset.drop('Gender',axis=1)
dataset=pd.concat([dataset,gender],axis=1)

#spliting data into two csv files
dt_training=dataset.sample(frac=0.7)
dt_testing=pd.concat([dataset,dt_training]).drop_duplicates(keep=False)
dt_training.to_csv('training_data.csv', header=True, index=None)
dt_testing.to_csv('test_data.csv', header=True, index=None)


#Save model
File_name='Randomforest.pkl'
pkl_file=open(File_name,'wb')
print("pickel file",pkl_file)
model=pickle.dump(classifier,pkl_file)



