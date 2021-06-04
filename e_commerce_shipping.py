#%% Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

import warnings 
warnings.filterwarnings('ignore')

import seaborn as sns
from collections import Counter
#%% Import Dataset

data = pd.read_csv("e_commerce_dataset.csv")

#%% Exploratory Data Analysis

data.shape

data.info() 

data.columns

# Reached.on.Time_Y.N is a target column on this dataset

data.rename({"Reached.on.Time_Y.N":"Target"},inplace = True,axis = 1)
data.drop(["ID"],inplace = True,axis = 1)
# Variable Description
"""
ID: ID Number of Customers.
Warehouse block: The Company have big Warehouse which is divided in to block such as A,B,C,D,E.
Mode of shipment:The Company Ships the products in multiple way such as Ship, Flight and Road.
Customer care calls: The number of calls made from enquiry for enquiry of the shipment.
Customer rating: The company has rated from every customer. 1 is the lowest (Worst), 5 is the highest (Best).
Cost of the product: Cost of the Product in US Dollars.
Prior purchases: The Number of Prior Purchase.
Product importance: The company has categorized the product in the various parameter such as low, medium, high.
Gender: Male and Female.
Discount offered: Discount offered on that specific product.
Weight in gms: It is the weight in grams.
Reached on time: It is the target variable, where 1 Indicates that the product has NOT reached on time and 0 indicates it has reached on time.
"""

# Correlation Matrix , which shows us a relationship between features (columns).
corr_matrix = data.corr()
sns.clustermap(corr_matrix,annot = True,fmt = ".2f")
plt.title("Correlation between Features")
plt.show()

#-------------------#
threshold = 0.3
filtre = np.abs(corr_matrix["Target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation Between Features w Corr Threshold 0.3")


# Categorical variables = Warehouse_block,Mode_of_Shipment,Product_importance ,Gender,Target

def bar_plot(variable):
    
    # get features
    var = data[variable]
    
    # count number of features
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (10,10))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{} : {}",variable,varValue)
    
categorical_variables = ["Warehouse_block","Mode_of_Shipment","Product_importance" ,"Gender","Target","Customer_care_calls","Prior_purchases"]
for Q in categorical_variables:
    bar_plot(Q)

# Numerical Variables = Cost_of_the_Product,Discount_offered,Weight_in_gms

def plot_hist(numerical_variable):
    plt.figure(figsize = (10,10))
    plt.hist(data[numerical_variable],bins = 1000,color = "green")
    plt.xlabel(numerical_variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist ".format(numerical_variable))
    plt.show()
    
numerical_variables = ["Cost_of_the_Product","Discount_offered","Weight_in_gms"]
for T in numerical_variables:
    plot_hist(T)
    
#%% Missing Values
data.columns[data.isnull().any()]
data.isnull().sum()

# Dataset has no any missing value.

#%% Outlier Detection

def detect_outliers(df,features):
    outlier_indices = []
    for c in features:
        # 1 st quartile
        Q1 = np.percentile(df[c],25)
        
        # 3 rd quartile
        Q3 = np.percentile(df[c],75)
        
        # IQR
        IQR = Q3 - Q1
        
        # Outlier step
        outlier_step = IQR * 1.5
   
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1-outlier_step) | (df[c] > Q3 + outlier_step)].index
        
        # store indeces
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers

data.loc[detect_outliers(data,["Cost_of_the_Product","Discount_offered","Weight_in_gms"])]

#%% Convert Categorical varaibles (as a string) to numerical

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

data["Warehouse_block"] = label_encoder.fit_transform(data["Warehouse_block"])
data["Mode_of_Shipment"] = label_encoder.fit_transform(data["Mode_of_Shipment"])
data["Product_importance"] = label_encoder.fit_transform(data["Product_importance"])
data["Gender"] = label_encoder.fit_transform(data["Gender"])

#%% Get X and Y Coordinates

y = data.Target.values
x_data = data.drop("Target",axis = 1)

#%% Normalization Operation

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

#%% Train-Test Split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


#%% Logistic Regression Classification

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("Acuracy of Logistic Regression : ",lr.score(x_test,y_test))

"""
Acuracy of Logistic Regression :  0.6377272727272727
"""

#%% K-Nearst Neighbor Classification

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
print("Accuracy of the KNN: ",knn.score(x_test,y_test))

# Let's find best k value for KNN

score_list = []
for each in range(1,100):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,100),score_list)
plt.title("K Value & Accuracy")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.show()

# best k value is 46

knn3 = KNeighborsClassifier(n_neighbors=46)
knn3.fit(x_train,y_train)
print("Accuracy of the KNN: ",knn3.score(x_test,y_test))

"""
Accuracy of the KNN:  0.6659090909090909
"""

#%% Support Vector Machines

from sklearn.svm import SVC
svm = SVC(random_state=1)
svm.fit(x_train,y_train)
print("Accuracy of the SVM: ",svm.score(x_test,y_test))

"""
Accuracy of the SVM:  0.6622727272727272
"""

#%% Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,random_state = 1)
rf.fit(x_train,y_train)
print("Accuracy of the RFC: ",rf.score(x_test,y_test))

"""
Accuracy of the RFC:  0.6636363636363637
"""

