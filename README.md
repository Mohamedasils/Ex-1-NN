<H3>ENTER YOUR NAME</H3>
<H3>ENTER YOUR REGISTER NO.</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
~~~
import pandas as pd                                                
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
~~~
~~~
df=pd.read_csv("Churn_Modelling.csv")
df
~~~
![Screenshot 2024-08-23 213816](https://github.com/user-attachments/assets/8c453a1e-6b58-4299-930e-9496b863d4aa)
~~~
df.isnull().sum()
~~~
![Screenshot 2024-08-23 214741](https://github.com/user-attachments/assets/be4bd280-3e35-49b7-b901-cbc3349292e8)
~~~
df.duplicated()
~~~
![Screenshot 2024-08-23 215113](https://github.com/user-attachments/assets/078a3603-18ba-4ead-bc9e-6cb40702af0f)
~~~
print(df['CreditScore'].describe())
~~~
![Screenshot 2024-08-23 215242](https://github.com/user-attachments/assets/9ef684c8-6608-44b2-af5b-2b8797987865)
~~~
df.info()
~~~
![image](https://github.com/user-attachments/assets/33d07704-6970-4589-b373-15dcdcc3fd34)
~~~
df.drop(['Surname','CustomerId','Geography','Gender'],axis=1,inplace=True)
df
~~~
![image](https://github.com/user-attachments/assets/8d561f83-db58-4607-ad9b-bb7f9040e8d8)
~~~
scaler=MinMaxScaler()
df=pd.DataFrame(scaler.fit_transform(df))
df
~~~
![Screenshot 2024-08-23 221238](https://github.com/user-attachments/assets/bca76260-721b-4544-abf3-c0de58767266)
~~~
X = df.iloc[:, :-1].values
print(X)
~~~
![Screenshot 2024-08-23 222644](https://github.com/user-attachments/assets/ac7253ae-e50e-42b4-9f09-8ce5a4a76a98)
~~~
y = df.iloc[:,-1].values
print(y)
~~~
![Screenshot 2024-08-23 224228](https://github.com/user-attachments/assets/588d924d-edc0-4a38-9c0b-b497c411033a)
~~~
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)
print(X_train)
print(len(X_train))
~~~
![Screenshot 2024-08-23 224228](https://github.com/user-attachments/assets/242eb640-83ee-43af-9287-cb13c4700e0a)
~~~
print(X_test)
print(len(X_test))
~~~
![image](https://github.com/user-attachments/assets/c7eb5132-3481-44f0-9abd-0756bcb49a45)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


