import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

df = pd.read_csv('weatherAUS.csv')
print(df.shape)
print(df.info())
print(df.isna().sum())
#drop columns
df = df.drop(["Evaporation","Sunshine","Cloud9am","Cloud3pm","Location","Date"], axis =1)
##print(df.describe());
df = df.dropna(axis = 0);
print(df.shape)
print(df.duplicated())
df.drop_duplicates()
from sklearn.preprocessing import LabelEncoder
#transform categorical data to numeric
labeler = LabelEncoder()
df['RainToday'] = labeler.fit_transform(df['RainToday'])
df['RainTomorrow'] = labeler.fit_transform(df['RainTomorrow'])
df['WindDir9am'] = labeler.fit_transform(df['WindDir9am'])
df['WindDir3pm'] = labeler.fit_transform(df['WindDir3pm'])
df['WindGustDir'] = labeler.fit_transform(df['WindGustDir'])

print(df.head())

#Detect outliars by using IQR
Q1=np.percentile(df['WindGustSpeed'], 75)
Q3=np.percentile(df['WindGustSpeed'], 25)

#IQR = Q3-Q1
IQR = df.WindGustSpeed.describe()['75%'] - df.WindGustSpeed.describe()['25%']
print ("WindGustSpeed IQR : ",IQR)

# Calculate the minimum value and maximum value
min = Q1-1.5*IQR
max = Q3+1.5*IQR
print ("minimum value: ",min)
print ("maximum value: ",max)

plt.boxplot(df.WindGustSpeed,notch=True,vert=False)
plt.show()


Humidity9am_mean=df.Humidity9am.mean()
Humidity3pm_mean=df.Humidity3pm.mean()
Humidity9am_median=df.Humidity9am.median()
Humidity3pm_median=df.Humidity3pm.median()
Humidity9am_std=df.Humidity9am.std()
Humidity3pm_std=df.Humidity3pm.std()
Humidity9am_var=df.Humidity9am.var()
Humidity3pm_var=df.Humidity3pm.var()

print ("Humidity9am_mean : ", Humidity9am_mean)
print ("Humidity3pm_mean : ", Humidity3pm_mean)
print ("Humidity3pm_median : ", Humidity3pm_median)
print ("Humidity9am_median : ", Humidity9am_median)
print ("Humidity9am std : ", Humidity9am_std)
print ("Humidity3pm_std : ", Humidity3pm_std)
print ("Humidity3pm_var  : ", Humidity3pm_var)
print ("Humidity9am var : ", Humidity9am_var)


X = df.drop(['RainTomorrow'], axis = 1)
Y = df['RainTomorrow']

##visualization##

#scatterplot matrix
df.hist(bins = 10 , figsize= (14,14))
plt.show()

#histogram for Maxtemp values
sns.histplot(x=df.MaxTemp)
plt.title("MaxTemp Distribution", color="red", fontsize=18)
plt.show()

# correlation heatmap
##using sns
# plt.figure(figsize=(8,8))
# sns.heatmap(df.corr())
# plt.show()


##using go
# Generate correlation matrix
df_corr = df.corr()
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x = df_corr.columns,
        y = df_corr.columns,
        z = np.array(df_corr)
    )
)
fig.show()

#scatterPlots
scatterPlot= px.scatter(df.sample(2000),
           title='Min Temp. vs Max Temp.',
           x='MinTemp',
           y='MaxTemp',
           color='RainToday')
scatterPlot.show()
#### It shows a linear positive correlation between minimum temperature and maximum temperature

scatterPlot= px.scatter(df.sample(2000),
           title='Humidity vs Temp.',
           x='Humidity3pm',
           y='Temp3pm',
           color='RainTomorrow')
scatterPlot.show()
#### It shows a linear negative correlation between humidity and  temperature

#barPlot
sns.barplot(data=df, x="RainTomorrow", y="Rainfall")
plt.show()
#### The higher the rate of rain, the greater the probability of rain tomorrow

#pie chart
fig = px.pie(df, names='RainToday', title='RainToday',color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,Y, test_size= 0.20 , random_state= 42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score

# Random forest
model_1=RandomForestClassifier(max_depth=7 , max_features=3,n_estimators= 100)
model_1.fit(x_train,y_train)
Random_forest_prediction = model_1.predict(x_test)

print( confusion_matrix(y_test,Random_forest_prediction))
print('randomForest accuracy ' ,accuracy_score(y_test,Random_forest_prediction))
print( 'randomForest recall ' ,recall_score(y_test,Random_forest_prediction))
print( 'randomForest precision ',precision_score(y_test,Random_forest_prediction))

#DecisionTreeClassifier
model_2 = DecisionTreeClassifier()
model_2.fit(x_train,y_train)
DecisionTree_prediction = model_2.predict(x_test)

print(confusion_matrix(y_test,DecisionTree_prediction))
print('decisionTree accuracy' ,accuracy_score(y_test,DecisionTree_prediction))
print('decisionTree recall' , recall_score(y_test,DecisionTree_prediction))
print('decisionTree precision',precision_score(y_test,DecisionTree_prediction))

#naive bayes classifier (Multinomial)
model_3 = MultinomialNB()
model_3.fit(x_train.abs(),y_train.abs())
Multinomial_prediction = model_3.predict(x_test)
print(confusion_matrix(y_test,Multinomial_prediction))
print('mulinomial accuracy' , accuracy_score(y_test,Multinomial_prediction))
print('mulinomial recall' , recall_score(y_test,Multinomial_prediction, average='weighted'))
print('mulinomial precision', precision_score(y_test,Multinomial_prediction, average='weighted'))

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train,y_train)
predictions = lr.predict(x_test)
print(confusion_matrix(y_test, predictions))
print('LogisticRegression accuracy' ,accuracy_score(y_test, predictions))

# knn
model_4 = KNeighborsClassifier(3)
model_4.fit(x_train,y_train)
knn_prediction = model_4.predict(x_test)

print(confusion_matrix(y_test,knn_prediction))
print('knn accuracy' , accuracy_score(y_test,knn_prediction))
print('knn recall' , recall_score(y_test, knn_prediction, average='weighted'))
print('knn precision', precision_score(y_test,knn_prediction, average='weighted'))
