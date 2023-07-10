#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # Step One: Frame the problem

# #The goal in this problem is to predict the average housing price for a given housing district, which will then be fed as an input to a more complex ML model that will be used to decide whether or not to invest in a given area or not. 

# #What type of ML model (batch vs online, supervised vs unsupervised, type of model, etc) do we need to perform this task? In this case, a linear regression model would work well since we need to know the labels of existing houses, and there is a general linear trend between housing features and housing prices. 

# #Select a performance measure or an evaluation metric. In this case we use root mean square error, which gives higher weight 
# #to larger errors. The link below is good for selecting the best performance measures 
# #https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide

# # Step Two: Download/Convert and Explore the Data

# In[2]:


import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[3]:


fetch_housing_data()


# In[4]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[5]:


housing = load_housing_data()


# In[6]:


housing.head()


# In[7]:


#We can observe one categorical column, so we have to use an imputer/encoder to vectorize that column 


# In[8]:


housing.info()   #info returns common df information and types
#total bedrooms is the only col with null values, maybe 


# In[9]:


def count_outliers(df, column_name):
    column = df[column_name]
    mean = column.mean()
    std = column.std()
    threshold = 2 * std
    lower_bound = mean - threshold
    upper_bound = mean + threshold
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return len(outliers)


# In[10]:


column_name = 'median_house_value'  # Replace 'column_name' with the actual column name
num_outliers = count_outliers(housing, column_name)
print(f"Number of outliers in '{column_name}': {num_outliers}")
#1383 is about 6 percent of the data, making it a reasonable number of outliers for the 2SD rule being used for outlier 
#identification


# In[11]:


print(len(housing))


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[13]:


#We see from the above graph a few interesting things.
#1: Median income in scaled in some way
#2: Median house value is capped at 50,000$, which could result in bad predictions. Need to fix this. Same problem with 
#housing_median_age. 


# # Step Three: Create the Test Set and/or Validation Sets

# random state parameter ensures that the same train and test sets are generated every execution to prevent model from seeing 
# new data every time

# Method One of Sampling: random sampling using train_test_split and random_state: Splits data randomly into 2 sets.

# In[14]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# Method Two of Sampling: Stratified Random Sampling, which makes sure that specific proportions are met in the sets which are representative of the population.
# 
# Use random sampling when you have large datasets and/or balanced data and/or having balanced data does not matter.
# Use stratified sampling when you have small datasets and/or unbalanced data and/or having balanced data does matter 
# In this case, income does matter a lot and small/large incomes could have a drastic impact on predictions, so use stratificationl

# In[15]:


housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])


# In[16]:


housing["income_cat"].hist()


# In[17]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[18]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[19]:


housing["income_cat"].value_counts() / len(housing)


# In[20]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# In[21]:


housing.head()


# # Step Four: Data Visualization

# In[22]:


housing = strat_train_set.copy()


# In[23]:


plt.scatter(housing['longitude'], housing['latitude'], alpha = 0.1)   #A map of california with more points in the urban areas


# In[24]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()


# In[25]:


# These plots reveal that location is a very important attribute, as well as distance to coast


# # Step Five: Feature Selection/Engineering

# There are numerous methods for feature selection (https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e) but in this case we are using the r correlation since there are not too many cols, as well as scatter_matrix. Large datasets would require other techniques for feature selection. 

# In[26]:


corr_matrix = housing.corr()


# In[27]:


corr_matrix['median_house_value'].sort_values()


# In[28]:


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize = (12,12))


# This corr chart shows that the most important related features by far is median_house_income and median_income. Let us look at that chart specifically

# In[29]:


plt.scatter(housing['median_income'], housing['median_house_value'], alpha=0.1)


# This corr chart shows numerous lines at 500,000, 450,000, 360,000, and a few below that. Not sure how that is happening, but might have to remove the houses at those points

# In[30]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[31]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[32]:


housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()


# # Step Six: Data Cleaning and Categorical Attributes 

# In[33]:


from sklearn.impute import SimpleImputer


# In[34]:


imputer = SimpleImputer(strategy = "median")


# In[35]:


housing_num = housing.drop("ocean_proximity", axis = 1)


# In[36]:


imputer.fit(housing_num)


# In[37]:


imputer.statistics_


# In[38]:


X = imputer.transform(housing_num)


# In[39]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


# In[40]:


housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


# In[41]:


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[42]:


from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[43]:


from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[119]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[45]:


from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# In[46]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[47]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))


# In[48]:


print("Labels:", list(some_labels))


# In[49]:


some_data_prepared


# In[50]:


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[51]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# In[52]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[53]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[54]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# In[55]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[132]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_searchs = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True, n_jobs=5)
grid_searchs.fit(housing_prepared, housing_labels)


#Grid search example, random search is similar, but it tries n random combos of hyperparameters for large search spaces


# # Question 1: Grid Search for SVR

# In[110]:


from sklearn.svm import SVR
SVR_model = SVR()
SVR_model.fit(housing_prepared, housing_labels)
risk_predictions = SVR_model.predict(housing_prepared)


# In[111]:


scores = cross_val_score(SVR_model, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
svr_rmse_scores = np.sqrt(-scores)


# In[112]:


display_scores(svr_rmse_scores)


# In[105]:


param_grid = [
    {"kernel":["rbf"], "C":[500,1000,1500,2003, 2500]}
]
grid_search = GridSearchCV(SVR_model, param_grid, cv = 5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs = 4)
grid_search.fit(housing_prepared, housing_labels)


# In[106]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[ ]:





# In[99]:


grid_search.best_params_


# In[100]:


SVR_model.get_params()


# # Question Two: Random Search on SVR

# In[113]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }


# In[114]:


import os
  
n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)


# In[115]:


random_search = RandomizedSearchCV(SVR_model, param_distribs, cv = 5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs = 5)
random_search.fit(housing_prepared, housing_labels)


# In[117]:


random_search_tuning_results = random_search.cv_results_
for mean_score, params in zip(random_search_tuning_results["mean_test_score"], random_search_tuning_results["params"]):
    print(np.sqrt(-mean_score), params)


# In[118]:


random_search.best_params_


# In[121]:


num_pipeline.get_params()


# # Question Three: Custom Transformer to select Top K features

# In[122]:


features = tree_reg.feature_importances_


# In[123]:


sorted_indices = np.argsort(features)


# In[128]:


sorted_indices


# In[133]:


feature_importances = grid_searchs.best_estimator_.feature_importances_
feature_importances


# In[134]:


cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attributes = list(cat_encoder.categories_[0])
attributes = num_attribs + cat_one_hot_attributes
sorted(zip(feature_importances, attributes))


# In[137]:


# custom transformer that selects only the top k most important attributes

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class selectTopKFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importance, k):
        self.feature_importance = feature_importance
        self.k = k
        
    def fit(self):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    
    def transform(self, X):
        return X[:, self.feature_indices_]


# In[ ]:




