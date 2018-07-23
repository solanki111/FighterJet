# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 10:04:33 2018

@author: Abhishek
"""
#%%
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split


data = pd.read_csv('C:/Users/../data.csv')

#Create a new function:
def num_missing(x):
    return sum(x.isnull())

#Applying per column:
print "Missing values per column:"
print data.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column

#Applying per row:
print "\nMissing values per row:"
print data.apply(num_missing, axis=1).head() #axis=1 defines that function is to be applied on each row
colnames = list(data)
array = data.values
X = array[:,0:137]
Y = array[:,137]

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

##
## Feature Extraction with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import Multi
# feature extraction
#model = LogisticRegression(multi_class='multinomial')
model = LogisticRegression()
model.fit(X_train, y_train)
print "Score:", model.score(X_test, y_test)
#accuracy = 0.854 for all predictors

predictions = model.predict(X_test)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)


rfe = RFE(model, 15)
rfefit = rfe.fit(X_train, y_train)
print("Num Features: %d") % rfefit.n_features_
print("Selected Features: %s") % rfefit.support_
print("Feature Ranking: %s") % rfefit.ranking_
print "Features sorted by their rank:"
print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), colnames))

# selecting 15 columns by name(indexes)
#'BlackBox R1', 'BlackBox R128', ' BlackBox R131', 'BlackBox R137', 'BlackBox R23', 'BlackBox R32', 'BlackBox R38', 'BlackBox R45', 'BlackBox R48', 'BlackBox R53', 'BlackBox R8', 'BlackBox R82', 'BlackBox R85', 'BlackBox R87', 'BlackBox R89'
X_train_new = X_train[:,(0, 127, 130, 136, 22, 31, 37, 44, 47, 52, 7, 81, 84, 86, 88)]
X_test_new = X_test[:,(0, 127, 130, 136, 22, 31, 37, 44, 47, 52, 7, 81, 84, 86, 88)]
model.fit(X_train_new, y_train)
print "Score:", model.score(X_test_new, y_test)
#accuracy = 0.849 for 15 predictors

rfe = RFE(model, 21)
rfefit = rfe.fit(X_train, y_train)
print("Num Features: %d") % rfefit.n_features_
print("Selected Features: %s") % rfefit.support_
print("Feature Ranking: %s") % rfefit.ranking_
print "Features sorted by their rank:"
print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), colnames))
# selecting 21 columns by name(indexes)
#'BlackBox R1', 'BlackBox R128', 'BlackBox R131', 'BlackBox R137', 'BlackBox R23', 'BlackBox R28', 'BlackBox R32', 'BlackBox R37', 'BlackBox R38', 'BlackBox R45', 'BlackBox R48', 'BlackBox R51', 'BlackBox R53', 'BlackBox R57', 'BlackBox R70', 'BlackBox R8', 'BlackBox R82', 'BlackBox R85', 'BlackBox R87', 'BlackBox R89', 'BlackBox R98'
X_train_new2 = X_train[:,(0, 127, 130, 136, 22, 27, 31, 36, 37, 44, 47, 50, 52, 56, 69, 7, 81, 84, 86, 88, 97)]
X_test_new2 = X_test[:,(0, 127, 130, 136, 22, 27, 31, 36, 37, 44, 47, 50, 52, 56, 69, 7, 81, 84, 86, 88, 97)]
model.fit(X_train_new2, y_train)
print "Score:", model.score(X_test_new2, y_test)
#accuracy = 0.854 for 21 predictors


## Performing PCA and modelling using random forest classifier 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train))
X_test_scaled = pd.DataFrame(scaler.transform(X_test))


# feature extraction
pca = PCA(n_components=20)
pcafit = pca.fit(X_train)
# summarize components
print("Explained Variance: %s") % pcafit.explained_variance_ratio_
print(pcafit.components_)


X_train_features = pca.fit_transform(X_train)
rfc = RandomForestClassifier(n_estimators = 100, n_jobs = 1, random_state = 2016, verbose = 1,                                      class_weight='balanced',oob_score=True)
rfc.fit(X_train_features, y_train)
X_test_features = pca.transform(X_test)
predicted = rfc.predict(X_test_features)
accuracy_score(y_test, predicted)
#accuracy 0.939

#%%
## Performing Data Imputation: Replacing 0 values by mean values for numeric columns #
# It can also replace the most frequent value for non-numeric columns but since we don't have that.
#

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
    
#%%    
X2 = pd.DataFrame(data)
X3 = DataFrameImputer().fit_transform(X2)

array = X3.values
X = array[:,0:137]
Y = array[:,137]

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
print "Score:", model.score(X_test, y_test)
#accuracy 0.856 for all  predictors

rfe = RFE(model, 21)
rfefit = rfe.fit(X_train, y_train)
print("Num Features: %d") % rfefit.n_features_
print("Selected Features: %s") % rfefit.support_
print("Feature Ranking: %s") % rfefit.ranking_
print "Features sorted by their rank:"
print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), colnames))
