# Telecom Churn:
The business objective is to predict the churn With provided predictor variables and recommend strategies to manage customer churn based on the observations.

## Data Preparation 

# Importing all the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils import resample
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, IncrementalPCA
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
# Importing required packages for visualization
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot,graphviz
from sklearn import metrics
from sklearn.svm import SVC
# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


import warnings
warnings.filterwarnings('ignore')

#Read the data from telecom_churn_data
churn = pd.read_csv('telecom_churn_data.csv')

churn.head()

churn.info()

churn.describe()

churn.shape

#Finding the columns having only one value
single_val_col = [] 
for x in churn.columns:
    if (churn[x].nunique()==1):
        single_val_col.append(x)
single_val_col

#Drop the single valued columns
churn_new = churn.drop(single_val_col,axis=1)  

#Dataframe after removing the single valued columns
churn_new.info()   

#Finding null values in the columns
round(churn_new.isnull().sum()/len(churn_new.index),2).sort_values(ascending=False)[:41]

#Columns having more than 70% null values
null_columns = (churn_new.isnull().sum()/len(churn_new.index)).sort_values(ascending=False)[:40].index
null_columns = sorted(null_columns)
np.array(null_columns)

#Impute the some important numeric columns with 0

col_impu_0 = ['av_rech_amt_data_6','av_rech_amt_data_7','av_rech_amt_data_8','av_rech_amt_data_9','total_rech_data_6',
              'total_rech_data_7','total_rech_data_8','total_rech_data_9','max_rech_data_6','max_rech_data_7',
              'max_rech_data_8','max_rech_data_9','date_of_last_rech_data_6','date_of_last_rech_data_7',
              'date_of_last_rech_data_8','date_of_last_rech_data_9','date_of_last_rech_6','date_of_last_rech_7',
              'date_of_last_rech_8','date_of_last_rech_9']
for x in col_impu_0:
    churn_new.loc[churn_new[x].isnull(),x]=0

#Impute the some categorical columns with -1

col_impu_1 = ['night_pck_user_6','night_pck_user_7','night_pck_user_8','night_pck_user_9','fb_user_6','fb_user_7',
              'fb_user_8','fb_user_9']
for x in col_impu_1:
    churn_new.loc[churn_new[x].isnull(),x]=-1;
churn_new[col_impu_1] = churn_new[col_impu_1].astype('object')
churn_new.info()

#Finding the remaining columns in 70% null value columns
null_columns = np.setdiff1d(null_columns,col_impu_0)
null_columns = np.setdiff1d(null_columns,col_impu_1)
null_columns

#Drop the remaining columns
churn_new = churn_new.drop(null_columns,axis=1)
churn_new.info()

#find again null value columns
round(churn_new.isnull().sum()/len(churn_new.index),2).sort_values(ascending=False)[:50]

#Imputing all the numeric columns with median those having less than 10% of null values  
des_col = churn_new.describe().columns
for x in des_col:
    churn_new.loc[churn_new[x].isnull(),x]=churn_new[x].median()

#Changing type of the variable from object to datetime

churn_new[['date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8','date_of_last_rech_9']] = churn_new[['date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8','date_of_last_rech_9']].apply(pd.to_datetime, errors='coerce')
churn_new[['date_of_last_rech_data_6','date_of_last_rech_data_7','date_of_last_rech_data_8','date_of_last_rech_data_9']] = churn_new[['date_of_last_rech_data_6','date_of_last_rech_data_7','date_of_last_rech_data_8','date_of_last_rech_data_9']].apply(pd.to_datetime, errors='coerce')

churn_new.info()

#Checking if is there any null values any
round(churn_new.isnull().sum()/len(churn_new.index),2).sort_values(ascending=False).head()

All important data quality checks are performed and inconsistent/missing data is handled appropriately.

### Hence data cleaning is finished 

## EDA

#Filtering only 6th month columns to understand the data
col_6 = churn_new.columns[churn_new.columns.str.contains('_6')]

churn_new[col_6[30:39]].head(10)

churn_new[col_6[39:]].head(10)

#Analysing columns to get high value customers
churn_new[['total_rech_data_6','max_rech_data_6','av_rech_amt_data_6']].sort_values(by='max_rech_data_6',
                                                                                    ascending=False).head(20)

#Defined a common function to generate boxplots for set of columns 
def EDA_plots_monthwise(columns_array):
    plt.figure(figsize=(20,12))
    num = 1
    for col in columns_array:
        plt.subplot(2,5,num)
        sns.boxplot(y=churn_new[col])
        plt.yscale('log')
        plt.title(col + " distribution")
        num = num+1
    plt.show()

#Univariate Analysis for 6th month features 
EDA_columns_6 = ['onnet_mou_6', 'offnet_mou_6', 'roam_ic_mou_6', 'roam_og_mou_6', 'total_og_mou_6', 'total_ic_mou_6', 'total_rech_num_6', 'total_rech_amt_6', 'av_rech_amt_data_6']
EDA_plots_monthwise(EDA_columns_6)    

#Univariate Analysis for 7th month features 
EDA_columns_7 = ['onnet_mou_7', 'offnet_mou_7', 'roam_ic_mou_7', 'roam_og_mou_7', 'total_og_mou_7', 'total_ic_mou_7', 'total_rech_num_7', 'total_rech_amt_7', 'av_rech_amt_data_7']
EDA_plots_monthwise(EDA_columns_7)

#Univariate Analysis for 8th month features 
EDA_columns_8 = ['onnet_mou_8', 'offnet_mou_8', 'roam_ic_mou_8', 'roam_og_mou_8', 'total_og_mou_8', 'total_ic_mou_8', 'total_rech_num_8', 'total_rech_amt_8', 'av_rech_amt_data_8']
EDA_plots_monthwise(EDA_columns_8)

#Univariate Analysis for 9th month features 
EDA_columns_9 = ['onnet_mou_9', 'offnet_mou_9', 'roam_ic_mou_9', 'roam_og_mou_9', 'total_og_mou_9', 'total_ic_mou_9', 'total_rech_num_9', 'total_rech_amt_9', 'av_rech_amt_data_9']
EDA_plots_monthwise(EDA_columns_9)

#Finding all the amount coloumns 
total_amt = churn_new.columns[churn_new.columns.str.contains('amt')]
total_amt

#Univariate Analysis for total amount related features 
EDA_amt_total = ['total_rech_amt_6', 'total_rech_amt_7', 'total_rech_amt_8', 'total_rech_amt_9']
EDA_plots_monthwise(EDA_amt_total)

#Univariate Analysis for average amount related features 
EDA_amt_avg = ['av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8', 'av_rech_amt_data_9']
EDA_plots_monthwise(EDA_amt_avg)

#Finding correlated columns 
churn_corr = churn_new.corr().rename_axis(None).rename_axis(None, axis=1)
churn_corr_max = churn_corr[(churn_corr > 0.9) & (churn_corr != 1)]
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                #colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add((corr_matrix.columns[i], corr_matrix.columns[j]))

    return col_corr
    
colss = correlation(churn_new, 0.9)
print(len(colss))
colss

## Derived Metrics

#total_og_mou_6,total_ic_mou_6,total_rech_amt_6,av_rech_amt_data_6
#Derived metrics

#Good Phase average variables
churn_new['avg_og_mou_6&7'] = (churn_new['total_og_mou_6']+churn_new['total_og_mou_7'])/2
churn_new['avg_ic_mou_6&7'] = (churn_new['total_ic_mou_6']+churn_new['total_ic_mou_7'])/2
churn_new['av_rech_amt_data_6&7'] = (churn_new['av_rech_amt_data_6']+churn_new['av_rech_amt_data_7'])/2
churn_new['avg_vol_2g_mb_6&7'] = (churn_new['vol_2g_mb_6']+churn_new['vol_2g_mb_7'])/2
churn_new['avg_vol_3g_mb_6&7'] = (churn_new['vol_3g_mb_6']+churn_new['vol_3g_mb_7'])/2
churn_new['avg_data_6&7'] = (churn_new['vol_3g_mb_6']+churn_new['vol_3g_mb_7']+
                               churn_new['vol_2g_mb_6']+churn_new['vol_2g_mb_7'])/2


#Difference b/w good phase months(i.e 6 and 7)
churn_new['diff_og_mou_6&7'] = (churn_new['total_og_mou_6']-churn_new['total_og_mou_7'])
churn_new['diff_ic_mou_6&7'] = (churn_new['total_ic_mou_6']-churn_new['total_ic_mou_7'])
churn_new['diff_rech_amt_data_6&7'] = (churn_new['av_rech_amt_data_6']-churn_new['av_rech_amt_data_7'])
churn_new['diff_rech_6&7'] = (churn_new['total_rech_amt_6']-churn_new['total_rech_amt_7']+
                             churn_new['av_rech_amt_data_6']-churn_new['av_rech_amt_data_7'])
churn_new['diff_vol_2g_mb_6&7'] = (churn_new['vol_2g_mb_6']-churn_new['vol_2g_mb_7'])
churn_new['dif_vol_3g_mb_6&7'] = (churn_new['vol_3g_mb_6']-churn_new['vol_3g_mb_7'])
churn_new['diff_data_6&7'] = (churn_new['vol_3g_mb_6']-churn_new['vol_3g_mb_7']+
                               churn_new['vol_2g_mb_6']-churn_new['vol_2g_mb_7'])

#Difference b/w good phase and bad phase
churn_new['diff_og_mou_7&8'] = (churn_new['avg_og_mou_6&7']-churn_new['total_og_mou_8'])
churn_new['diff_ic_mou_7&8'] = (churn_new['avg_ic_mou_6&7']-churn_new['total_ic_mou_8'])
churn_new['diff_rech_amt_data_7&8'] = (churn_new['av_rech_amt_data_6&7']-churn_new['av_rech_amt_data_8'])
churn_new['diff_rech_7&8'] = (churn_new['total_rech_amt_7']+churn_new['total_rech_amt_6']-2*churn_new['total_rech_amt_8']+
                             churn_new['av_rech_amt_data_7']+churn_new['av_rech_amt_data_6']-2*churn_new['av_rech_amt_data_8'])/2
churn_new['diff_vol_2g_mb_7&8'] = (churn_new['avg_vol_2g_mb_6&7']-churn_new['vol_2g_mb_8'])
churn_new['dif_vol_3g_mb_7&8'] = (churn_new['avg_vol_3g_mb_6&7']-churn_new['vol_3g_mb_8'])
churn_new['diff_data_7&8'] = (churn_new['avg_data_6&7']-churn_new['vol_3g_mb_8']-churn_new['vol_2g_mb_7'])

#Bad Phase total variables
churn_new['avg_rech_8'] = (churn_new['total_rech_amt_8']+churn_new['av_rech_amt_data_7'])
churn_new['total_data_8'] = (churn_new['vol_2g_mb_8']+churn_new['vol_3g_mb_8'])

## Filtering high-value customers

churn_new['avg_rech_6&7'] = (churn_new['total_rech_amt_6']+churn_new['total_rech_amt_7']+
                             churn_new['av_rech_amt_data_6']+churn_new['av_rech_amt_data_7'])/2

churn_new['avg_rech_6&7'].describe(percentiles=[0.7])

high_val_cost = churn_new[churn_new['avg_rech_6&7']>=431]
high_val_cost.info()



## Tagging churned customers

#total_ic_mou_9,total_og_mou_9,vol_2g_mb_9,vol_3g_mb_9
high_val_cost['churn_mou']  = (high_val_cost['total_ic_mou_9']+high_val_cost['total_og_mou_9'])
high_val_cost['churn_data'] =  (high_val_cost['vol_2g_mb_9']+high_val_cost['vol_3g_mb_9'])

high_val_cost['churn'] = (high_val_cost['churn_mou']>0) | (high_val_cost['churn_data']>0)

high_val_cost[['total_ic_mou_9','total_og_mou_9','vol_2g_mb_9','vol_3g_mb_9','churn']].head(10)

high_val_cost['churn'] = high_val_cost['churn'].apply(lambda x: 0 if (x==True) else 1) #Tag the churn as 1 else 0

#Checking the tag is correct or not
high_val_cost[['total_ic_mou_9','total_og_mou_9','vol_2g_mb_9','vol_3g_mb_9','churn']].head() 

high_val_cost.churn.sum()  # total no of churn in the high value customers

high_val_cost.churn.count()  # total no of churn in the high value customers

#### Churn Customers count: 2460
#### Non Churn Customers count: 27555

#Removing the churn month columns
high_val_cost = high_val_cost.drop(['churn_mou','churn_data'] ,axis=1)
col_9th = churn_new.columns[churn_new.columns.str.contains('_9')]
high_val_cost = high_val_cost.drop(col_9th,axis=1)  

high_val_cost.info()  #Dataframe after dropping churn month columns

#Finding years which are present
for x in ['date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8']:
    x = churn_new[x].apply(lambda a:a.year if a != 0 else 0)
    print (x.value_counts())

#Removing 1970 data which are very lesss
high_val_cost = high_val_cost[high_val_cost['date_of_last_rech_6'].apply(lambda a:a.year if a != 0 else 0)!=1970]

for x in ['date_of_last_rech_6','date_of_last_rech_6','date_of_last_rech_6']:
    x = high_val_cost[x].apply(lambda a:a.year if a != 0 else 0)
    print (x.value_counts())

# Creating individual columns date,month,year for all the date_of_last_rech_data columns
high_val_cost['day_of_last_rech_data_6'] = high_val_cost['date_of_last_rech_data_6'].apply(lambda x: x.day if x !=0 else 0).astype(int)
high_val_cost['day_of_last_rech_data_7'] = high_val_cost['date_of_last_rech_data_7'].apply(lambda x: x.day if x !=0 else 0).astype(int)
high_val_cost['day_of_last_rech_data_8'] = high_val_cost['date_of_last_rech_data_8'].apply(lambda x: x.day if x !=0 else 0).astype(int)

# Creating individual columns date,month,year for all the date_of_last_rech columns
high_val_cost['day_of_last_rech_6'] = high_val_cost['date_of_last_rech_6'].apply(lambda x: x.day if x !=0 else 0).astype(int)
high_val_cost['day_of_last_rech_7'] = high_val_cost['date_of_last_rech_7'].apply(lambda x: x.day if x !=0 else 0).astype(int)
high_val_cost['day_of_last_rech_8'] = high_val_cost['date_of_last_rech_8'].apply(lambda x: x.day if x !=0 else 0).astype(int)

#Dropping date_of_last_rech_data columns
high_val_cost.drop(['date_of_last_rech_data_6','date_of_last_rech_data_7','date_of_last_rech_data_8'],axis=1,inplace=True)
high_val_cost.drop(['date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8'],axis=1,inplace=True)

high_val_cost.info()

## EDA on High Valued Customers

#Finding important columns to do EDA
high_val_cost.columns[high_val_cost.columns.str.contains('_6')]

#Good Phase Plots
plt.figure(1,figsize=(20,12))
plt.subplot(231)
sns.boxplot(y='avg_og_mou_6&7',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Average Outgoing MOU vs Churn')
plt.subplot(232)
sns.boxplot(y='avg_ic_mou_6&7',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Average Incoming MOU vs Churn')
plt.subplot(233)
sns.boxplot(y='av_rech_amt_data_6&7',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Average recharge vs Churn')
plt.subplot(234)
sns.boxplot(y='avg_vol_2g_mb_6&7',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Average 2G data vs churn')
plt.subplot(235)
sns.boxplot(y='avg_vol_3g_mb_6&7',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Average 3G data vs churn')
plt.subplot(236)
sns.boxplot(y='avg_data_6&7',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Average 3G data vs churn')
plt.show()

#Bad Phase Plots
plt.figure(1,figsize=(20,12))
plt.subplot(231)
sns.boxplot(y='total_og_mou_8',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Average Outgoing MOU vs Churn')
plt.subplot(232)
sns.boxplot(y='total_ic_mou_8',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Average Incoming MOU vs Churn')
plt.subplot(233)
sns.boxplot(y='avg_rech_8',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Average recharge vs Churn')
plt.subplot(234)
sns.boxplot(y=high_val_cost['vol_2g_mb_8'],x=high_val_cost['churn'])
plt.yscale('log')
plt.title('Average 2G data vs churn')
plt.subplot(235)
sns.boxplot(y=high_val_cost['vol_3g_mb_8'],x=high_val_cost['churn'])
plt.yscale('log')
plt.title('Average 3G data vs churn')
plt.subplot(236)
sns.boxplot(y=high_val_cost['total_data_8'],x=high_val_cost['churn'])
plt.yscale('log')
plt.title('Total Data vs churn')
plt.show()

#Difference between good and bad phase
plt.figure(1,figsize=(20,12))
plt.subplot(231)
sns.boxplot(y='diff_og_mou_7&8',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Difference Outgoing MOU vs Churn')
plt.subplot(232)
sns.boxplot(y='diff_ic_mou_7&8',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Difference Incoming MOU vs Churn')
plt.subplot(233)
sns.boxplot(y='diff_rech_7&8',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Difference recharge vs Churn')
plt.subplot(234)
sns.boxplot(y='diff_data_7&8',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Average 3G data vs churn')
plt.subplot(235)
sns.boxplot(y='diff_vol_2g_mb_7&8',x='churn',data=high_val_cost)
plt.title('Difference 2G data vs churn')
plt.yscale('log')
plt.subplot(236)
sns.boxplot(y='dif_vol_3g_mb_7&8',x='churn',data=high_val_cost)
plt.yscale('log')
plt.title('Difference 3G data vs churn')
plt.show()

### Insights from EDA:

I) Good Phase(average of 6&7 months): Average 3g data for the churned customers is having high values compared to non churned, so Average 3g data is the main insight from good phase to find churned customers.


II) Bad Phase(8 month data): Average recharge for the month 8th is having mean around 0 compared to non churned which is having around 1000, so if the recharge is 0 then there are high chances of churned. 
Same case with average 2g and 3g data


III) Difference b/w good phase and bad phase: 
Difference b/w outgoing MOU of bad phase and good phase mean is around 0 which is very high for churned customers, same case with incoming mou and recharge. 
So if the mean is around 0 for the above mentioned values then more chances of churned.

## Handling Class Imbalance

churn_class = high_val_cost[high_val_cost['churn']==1]
nonchurn_class = high_val_cost[high_val_cost['churn']==0]


#down sample non churn data
nonchurn_sam = resample(nonchurn_class,replace=False,n_samples=3*len(churn_class),random_state =120)

#up sample churn class
churn_sam = resample(churn_class,replace=True,n_samples=3*len(churn_class),random_state=120)

bal_churn_data = pd.concat([nonchurn_sam,churn_sam])

## Dimensionality reduction using PCA

high_val_pca = bal_churn_data.copy()

#Processing categorical variables for PCA
high_val_pca.loc[high_val_pca.fb_user_6 == -1, 'fb_user_6'] = (len(high_val_pca[(high_val_pca.fb_user_6 == -1) & (high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.loc[high_val_pca.fb_user_6 == 1, 'fb_user_6'] = (len(high_val_pca[(high_val_pca.fb_user_6 == 1) & (high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.loc[high_val_pca.fb_user_6 == 0, 'fb_user_6'] = (len(high_val_pca[(high_val_pca.fb_user_6 == 0) & (high_val_pca.churn == 1)])/len(high_val_pca))
print(high_val_pca.fb_user_6.value_counts())

high_val_pca.loc[high_val_pca.fb_user_7 == -1,'fb_user_7'] = (len(high_val_pca[(high_val_pca.fb_user_7== -1) &(high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.loc[high_val_pca.fb_user_7 == 1, 'fb_user_7'] = (len(high_val_pca[(high_val_pca.fb_user_7== 1) & (high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.loc[high_val_pca.fb_user_7 == 0, 'fb_user_7'] = (len(high_val_pca[(high_val_pca.fb_user_7== 0) & (high_val_pca.churn == 1)])/len(high_val_pca))
print(high_val_pca.fb_user_7.value_counts())

high_val_pca.loc[high_val_pca.fb_user_8 == -1,'fb_user_8'] = (len(high_val_pca[(high_val_pca.fb_user_8== -1) &(high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.loc[high_val_pca.fb_user_8 == 1, 'fb_user_8'] = (len(high_val_pca[(high_val_pca.fb_user_8== 1) & (high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.loc[high_val_pca.fb_user_8 == 0, 'fb_user_8'] = (len(high_val_pca[(high_val_pca.fb_user_8== 0) & (high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.fb_user_8.value_counts()

high_val_pca.loc[high_val_pca.night_pck_user_6 == -1,'night_pck_user_6'] = (len(high_val_pca[(high_val_pca.night_pck_user_6 == -1) &(high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.loc[high_val_pca.night_pck_user_6 == 1, 'night_pck_user_6'] = (len(high_val_pca[(high_val_pca.night_pck_user_6 == 1) & (high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.loc[high_val_pca.night_pck_user_6 == 0, 'night_pck_user_6'] = (len(high_val_pca[(high_val_pca.night_pck_user_6 == 0) & (high_val_pca.churn == 1)])/len(high_val_pca))
print(high_val_pca.night_pck_user_6.value_counts())

high_val_pca.loc[high_val_pca.night_pck_user_7 == -1,'night_pck_user_7'] = (len(high_val_pca[(high_val_pca.night_pck_user_7 == -1) &(high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.loc[high_val_pca.night_pck_user_7 == 1, 'night_pck_user_7'] = (len(high_val_pca[(high_val_pca.night_pck_user_7 == 1) & (high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.loc[high_val_pca.night_pck_user_7 == 0, 'night_pck_user_7'] = (len(high_val_pca[(high_val_pca.night_pck_user_7 == 0) & (high_val_pca.churn == 1)])/len(high_val_pca))
print(high_val_cost.night_pck_user_7.value_counts())

high_val_pca.loc[high_val_pca.night_pck_user_8 == -1,'night_pck_user_8'] = (len(high_val_pca[(high_val_pca.night_pck_user_8 == -1) &(high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.loc[high_val_pca.night_pck_user_8 == 1, 'night_pck_user_8'] = (len(high_val_pca[(high_val_pca.night_pck_user_8 == 1) & (high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.loc[high_val_pca.night_pck_user_8 == 0, 'night_pck_user_8'] = (len(high_val_pca[(high_val_pca.night_pck_user_8 == 0) & (high_val_pca.churn == 1)])/len(high_val_pca))
high_val_pca.night_pck_user_8.value_counts()

high_val_pca.info()

# Separating Independent and dependent variables to go forward for train test split of the data

# Putting feature variable to X
X_PCA = high_val_pca.drop(['churn','mobile_number'],axis=1)

# Putting response variable to y
y_pca = high_val_pca['churn']

#Splitting the data into train and test using train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)

#y_reshape = np.array(y_train).reshape(-1,1)
#y_test_reshape = np.array(y_test).reshape(-1,1)

#Normalisation of data
X_train_norm = normalize(X_PCA)

pca_nrm = PCA(svd_solver='randomized', random_state=42)
pca_nrm.fit(X_train_norm)

#Dominent columns in PC1
colnames_nrm = list(X_PCA.columns)
pca1_df = pd.DataFrame({'Feature':colnames_nrm,'PC1':pca_nrm.components_[0],'PC2':pca_nrm.components_[1]})
pca1_df.sort_values('PC1',ascending=False).head(10)

#Dominent columns in PC2
pca1_df.sort_values('PC2',ascending=False).head(15)

#Making the screeplot - plotting the cumulative variance against the number of components
%matplotlib inline
fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca_nrm.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

#Variance of each PC 
pca_nrm.explained_variance_ratio_[:10]

# Business Understanding

## Logistic Regression

high_val_cost_lr = bal_churn_data.copy()

X = high_val_cost_lr.drop(['churn','mobile_number'],axis=1)

# Putting response variable to y
y = high_val_pca['churn']

x_log = pd.get_dummies(X,drop_first = True)

x_log.info()

#Splitting the data into train and test using train_test_split
x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(x_log,y, train_size=0.7,test_size=0.3,random_state=100)

x_train_sm = sm.add_constant(x_train_log)

lr = sm.GLM(y_train_log,x_train_sm).fit()
print (lr.summary())

lr1 = LogisticRegression()

rfe = RFE(lr1, 10)

#rfe = rfe.fit(x_train_log, y_train_log)
#rfe_columns = x_train_log.columns[rfe.support_]

rfe_columns = ['total_rech_num_8', 'total_rech_data_8', 'monthly_2g_7', 'monthly_2g_8','sachet_2g_6', 'sachet_2g_8',
               'monthly_3g_8', 'sachet_3g_8','day_of_last_rech_8', 'fb_user_8_0.0']

lr1.fit(x_train_log[rfe_columns],y_train_log)
print (lr1.coef_)
print (lr1.intercept_)

### Checking VIF

# UDF for calculating vif value
def vif_cal(input_data, dependent_col):
    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.OLS(y,x).fit().rsquared  
        vif=round(1/(1-rsq),2)
        vif_df.loc[i] = [xvar_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)

# vif_df = pd.get_dummies(high_val_cost_lr,drop_first = True)
# ref_col = vif_df[rfe_columns]
# vif_cal(input_data=ref_col, dependent_col="churn")

y_pred_log = lr1.predict(x_test_log[rfe_columns])
confusion_matrix(y_test_log, y_pred_log)
#y_pred_log

print('Recall Score:', recall_score(y_test_log, y_pred_log))
print('Precision Score:', precision_score(y_test_log, y_pred_log))
print('Accuracy Score:', accuracy_score(y_test_log, y_pred_log))

#### EDA based on Logistic Regression

#Created bins on last recharge day of 8th month to get the insights 
bins = [0, 5, 10, 20,25,31]
high_val_cost_lr['day_of_last_rech_8_bin'] = pd.cut(high_val_cost_lr['day_of_last_rech_8'], bins)
sns.countplot(x='day_of_last_rech_8_bin',hue='churn',data=high_val_cost_lr)
plt.show()

#Created bins on total recharge number of 8th month to get the insights 
bins = [0, 5, 10, 25, 50, 100,200]
high_val_cost_lr['total_rech_num_8_bin'] = pd.cut(high_val_cost_lr['total_rech_num_8'], bins)
sns.countplot(x='total_rech_num_8_bin',hue ='churn',data=high_val_cost_lr)
plt.show()

#Created bins on fb user of 8th month to get the insights 
sns.countplot(x='fb_user_8',hue ='churn',data=bal_churn_data)
plt.show()

#Comparing total recharge data around 0 to get the insights 
data_8_0 = high_val_cost_lr[high_val_cost_lr['total_rech_data_8']<=0]
data_8_1 = high_val_cost_lr[high_val_cost_lr['total_rech_data_8']>0]
plt.figure(figsize=(10,4))
plt.subplot(121)
sns.countplot(x='churn',data=data_8_0)
plt.title('Total_rech_data_8<=0')
plt.subplot(122)
sns.countplot(x='churn',data=data_8_1)
plt.title('Total_rech_data_8>0')
plt.show()

#Comparing total sachet 2g data around 0 to get the insights 
sachet_2g_8_0 = high_val_cost_lr[high_val_cost_lr['sachet_2g_8']<=0]
sachet_2g_8_1 = high_val_cost_lr[high_val_cost_lr['sachet_2g_8']>0]
plt.figure(figsize=(10,4))
plt.subplot(121)
sns.countplot(x='churn',data=sachet_2g_8_0)
plt.title('sachet_2g_8<=0')
plt.subplot(122)
sns.countplot(x='churn',data=sachet_2g_8_1)
plt.title('sachet_2g_8>0')
plt.show()

#Counting monthly 2g data for the month of 7 to draw instances
plt.figure(figsize=(20,12))
sns.countplot(x='monthly_2g_7',hue ='churn',data=bal_churn_data)
plt.show()

## Decision Tree

high_val_cost_dt = bal_churn_data.copy()

high_val_cost_dt.info()

# Putting feature variable to X
X = high_val_cost_dt.drop(['mobile_number','churn'],axis=1)

# Putting response variable to y
y = high_val_cost_dt['churn']

#Splitting the data into train and test using train_test_split
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)

#### taking max_depth as 5 for business understanding.

# Fitting the decision tree with default hyperparameters, apart from

dt_default = DecisionTreeClassifier(max_depth=5,random_state=42)
dt_default.fit(X_train_dt, y_train_dt)

y_pred_dt = dt_default.predict(X_test_dt)

confusion_matrix(y_test_dt, y_pred_dt)

#Metrics from the Desicion tree built
print('Recall Score:', recall_score(y_test_dt, y_pred_dt))
print('Precision Score:', precision_score(y_test_dt, y_pred_dt))
print('Accuracy Score:', accuracy_score(y_test_dt, y_pred_dt))


# Putting features
features = list(X.columns[0:])

# If you're on windows:
# Specifing path for dot file.
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# # plotting tree with max_depth=3
# dot_data = StringIO()  
# export_graphviz(dt_default, out_file=dot_data,
#                 feature_names=features, filled=True,rounded=True)

# graph = pydot.graph_from_dot_data(dot_data.getvalue())  
# graph[0].write_pdf("churn_PCA_df.pdf")

# plotting tree with max_depth=3
dot_data = StringIO()  
export_graphviz(dt_default, out_file=dot_data,
                feature_names=features, filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())

### EDA based on Decision Trees

#Case1: Place these constrains on data: 1. total_ic_mou_8     <=  30.02
#                                       2. total_data_8       <=  35.018
#                                       3. total_og_mou_8     <=  0.22
#                                       4. vol_3g_mb_7        <=  3.5

churn_case1 = high_val_cost_dt[(high_val_cost_dt['total_ic_mou_8']<=30.02) & (high_val_cost_dt['total_data_8']<=35.015)]
churn_case1 = churn_case1[(churn_case1['total_og_mou_8']<=0.22)&(churn_case1['vol_3g_mb_7']<=0.22)]
sns.countplot(x='churn',data=churn_case1)
plt.title('Churn Case_1')
plt.show()

#### Above filtered data churn percentage is 96.74%.

#Case2: Place these constrains on data: 1. total_ic_mou_8     <=  30.02
#                                       2. total_data_8       <=  35.018
#                                       3. total_og_mou_8     >   0.22
#                                       4. total_og_mou_8     <=  453.405
#                                       5. last_day_rch_amt_8 <=  3.5

churn_case2 = high_val_cost_dt[(high_val_cost_dt['total_ic_mou_8']<=30.02) & (high_val_cost_dt['total_data_8']<=35.018)]
churn_case2 = churn_case2[(churn_case2['total_og_mou_8']>0.22)&(churn_case2['total_og_mou_8']<=453.405) &(churn_case2['last_day_rch_amt_8']<=3.5)]
sns.countplot(x='churn',data=churn_case2)
plt.title('Churn Case_2')
plt.show()

#### Above filtered data churn percentage is 89.56%.

#Case3: Place these constrains on data: 1. total_ic_mou_8     >   30.02
#                                       2. roam_og_mou_8      >   0.005
#                                       3. loc_ic_mou_8       <=  191.36
#                                       4. roam_og_mou_7      <=  1.005
#                                       5. loc_og_t2f_mou_6   <=  0.39


churn_case3 = high_val_cost_dt[(high_val_cost_dt['total_ic_mou_8']>30.02)&(high_val_cost_dt['roam_og_mou_8']>0.005)]
churn_case3 = churn_case3[(churn_case3['loc_ic_mou_8']<=191.36)&(churn_case3['roam_og_mou_7']<=1.005)&(churn_case3['loc_og_t2f_mou_6']<=0.39)]
sns.countplot(x='churn',data=churn_case3)
plt.title('Churn Case_3')
plt.show()

#### Above filtered data churn percentage is 84.09%.

### Decision Tree for Modelling

# Fitting the decision tree with default hyperparameters, apart from

model_dt_default = DecisionTreeClassifier(max_depth=25,random_state=42)
model_dt_default.fit(X_train_dt, y_train_dt)

y_pred_dt_model = model_dt_default.predict(X_test_dt)

confusion_matrix(y_test_dt, y_pred_dt_model)

#Metrics from the Desicion tree built
print('Recall Score:', recall_score(y_test_dt, y_pred_dt_model))
print('Precision Score:', precision_score(y_test_dt, y_pred_dt_model))
print('Accuracy Score:', accuracy_score(y_test_dt, y_pred_dt_model))

## Model building with PCA

#Splitting the data into train and test using train_test_split
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_PCA, y_pca, train_size=0.7,test_size=0.3,random_state=100)

#Doing IncrementalPCA by taking 32 components which given more than 90% of data
pca_final = IncrementalPCA(n_components=32)

# Converting data to Principle components
df_train_pca = pca_final.fit_transform(X_train_pca)
df_test_pca = pca_final.transform(X_test_pca)

### Logistic Regression

#Defining different variables for Logistic PCA
X_train_pca_log = X_train_pca
X_test_pca_log = X_test_pca

#Fitting the PCA data to LogisticRegression
log_pca = LogisticRegression(random_state=42)
log_pca.fit(X_train_pca_log, y_train_pca)

#Predecting the output values
y_pred_pca_log = log_pca.predict(X_test_pca)

y_pred_pca_log

#confusion matrix
confusion_matrix(y_test_pca, y_pred_pca_log)

#Model metrics
print('Recall Score:', recall_score(y_test_pca, y_pred_pca_log))
print('Precision Score:', precision_score(y_test_pca, y_pred_pca_log))
print('Accuracy Score:', accuracy_score(y_test_pca, y_pred_pca_log))

#### Comparing logistic business metric with and without PCA
Logistic without PCA Recall Score: 0.8010018214936248 

Logistic with PCA Recall Score: 0.8442622950819673

### hyper parameter tuning

params = {'C': [0.1, 80, 90, 100, 110, 120, 500, 1000], 'penalty':['l1', 'l2'] }
log_tune = LogisticRegression(random_state=42)
folds = KFold(n_splits=5, shuffle=True, random_state=42)
clf = GridSearchCV(estimator=log_tune, param_grid=params, scoring='recall', cv=folds, verbose=1, return_train_score=True)

model_log_tuned_cv = clf.fit(X_train_pca_log, y_train_pca)

#Finding best parameters from best_estimator_
print('Best Penalty:', model_log_tuned_cv.best_estimator_.get_params()['penalty'])
print('Best C:', model_log_tuned_cv.best_estimator_.get_params()['C'])

#Fitting the model 
y_pred_log_cv = model_log_tuned_cv.predict(X_test_pca)

#Confusion Matrix
confusion_matrix(y_test_pca,y_pred_log_cv)

#Model metrics
recall_score(y_test_pca, y_pred_log_cv)
print('Recall Score:', recall_score(y_test_pca, y_pred_log_cv))
print('Precision Score:', precision_score(y_test_pca, y_pred_log_cv))
print('Accuracy Score:', accuracy_score(y_test_pca, y_pred_log_cv))

#### Comparing logistic business metric with and without PCA
Logistic without PCA Recall Score: 0.8010018214936248 

Logistic with PCA Recall Score: 0.8442622950819673

Logistic After hyper parameter tuning Recall Score: 0.8542805100182149

### SVM

#### Since SVM taking huge time code has been commented out by mentioning output as comment

#Model building using SVM
model_SVM_PCA = SVC(C=1, random_state=42, kernel='poly')

#Fit the train data
#model_SVM_PCA.fit(df_train_pca, y_train_pca)

#Output of fit
#SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='poly', max_iter=-1, probability=False, random_state=42, shrinking=True, tol=0.001, verbose=False)

# predict output values
#y_pred_svm = model_SVM_PCA.predict(df_test_pca)

#Confusion Matrix
#confusion_matrix(y_test_pca,y_pred_svm)

#Confusion matrix output
#array([[2214, 0], 
#       [ 264, 1950]], dtype=int64)

#Model Metrics
# accuracy
# print("Accuracy Score:", accuracy_score(y_test_pca, y_pred_svm))

# # precision
# print("Precision Score:", precision_score(y_test_pca, y_pred_svm))

# # recall/sensitivity
# print("Recall Score:", recall_score(y_test_pca, y_pred_svm))

#Metrics Output 
#Accuracy Score: 0.940379403794038 
#Precision Score: 1.0 
#Recall Score: 0.8807588075880759

### Hyper patameter tuning for SVM

#Tuning hyper parameter C of SVM
params = {"C": [0.1, 1, 10, 100, 1000]}
model_SVM_PCA_tuned = SVC()
folds = KFold(n_splits=5, random_state=42)
model_svm_cv = GridSearchCV(estimator=model_SVM_PCA_tuned, param_grid=params, scoring='recall', cv=folds, verbose=1, return_train_score=True)

#fit the training data on SVM
#model_svm_cv.fit(df_train_pca, y_train_pca)

#Output for fit
#Fitting 5 folds for each of 5 candidates, totalling 25 fits 
#[Parallel(n_jobs=1)]: Done 25 out of 25 | elapsed: 6.7min finished 

#GridSearchCV(cv=KFold(n_splits=5, random_state=42, shuffle=False), error_score='raise', estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False), fit_params=None, iid=True, n_jobs=1, param_grid={'C': [0.1, 1, 10, 100, 1000]}, pre_dispatch='2*n_jobs', refit=True, return_train_score=True, scoring='accuracy', verbose=1

#Predicting on test data
#y_pred_svm_tuned = model_svm_cv.predict(df_test_pca)

#confusion matrix
#confusion_matrix(y_test_pca,y_pred_svm_tuned)

#Confusion matrix Output
#array([[2214, 0], 
#       [ 264, 1950]], dtype=int64)

# print('Recall Score:', recall_score(y_test_pca,y_pred_svm_tuned))
# print('Precision Score:', precision_score(y_test_pca,y_pred_svm_tuned))
# print('Accuracy Score:', accuracy_score(y_test_pca,y_pred_svm_tuned))

#### Comparing SVM business metrics with and without hyper parameter tuning on PCA data
SVM without Tuning Recall Score:  0.8807588075880759

SVM After hyper parameter tuning Recall Score: 0.8807588075880759

### Decision Tree

#Building a model using Decision Tree
# Fitting the decision tree with max_depth equal to 10 on PCA data
model_dt_pca = DecisionTreeClassifier(max_depth=10,random_state=42)
model_dt_pca.fit(df_train_pca, y_train_pca)

#Predicting the output variable 
y_pred_dt_pca = model_dt_pca.predict(df_test_pca)

#Confusion Matrix
confusion_matrix(y_test_pca, y_pred_dt_pca)

#Model Metrics
print('Recall Score:', recall_score(y_test_pca, y_pred_dt_pca))
print('Precision Score:', precision_score(y_test_pca, y_pred_dt_pca))
print('Accuracy Score:', accuracy_score(y_test_pca, y_pred_dt_pca))

#### Comparing Decision Tree business metric with and without PCA
Decision Tree without PCA Recall Score: 0.9658469945355191 

Decision Tree with PCA Recall Score: 0.8970856102003643

## Hyper parameters tuning for DT

#tuning hyper parameters 
params = {
    'max_depth': range(15,50,5),
    'min_samples_leaf': range(1,11,2),
    'min_samples_split': range(2,10,2),
    'criterion': ['entropy','gini']
}

#Building model for hyper parameter tuning
model_tuned_dt = DecisionTreeClassifier(random_state=42)

model_tuned_dt_cv = GridSearchCV(estimator=model_tuned_dt, param_grid=params,cv=folds, scoring='recall', verbose=1,return_train_score=True)
model_tuned_dt_cv.fit(df_train_pca, y_train_pca)

#Checking best parameters using  best_estimator_
model_tuned_dt_cv.best_estimator_

#Predicting on test data
y_pred_dt_tuned = model_tuned_dt_cv.predict(df_test_pca)

#Confusion Matrix
confusion_matrix(y_test_pca, y_pred_dt_tuned)

#Model Metrics
print('Recall Score:', recall_score(y_test_pca, y_pred_dt_tuned))
print('Precision Score:', precision_score(y_test_pca, y_pred_dt_tuned))
print('Accuracy Score:', accuracy_score(y_test_pca, y_pred_dt_tuned))

#### Comparing Decision Tree business metric with and without PCA after hyper parameter tuning
Decision Tree without PCA Recall Score: 0.9658469945355191 

Decision Tree with PCA Recall Score: 0.8970856102003643

Decision Tree After hyper parameter tuning Recall Score: 0.9503642987249544

### Random Forest

# Building the random forest with default parameters.
model_rfc_pca = RandomForestClassifier(max_depth=10,random_state=100)

# fit
model_rfc_pca.fit(df_train_pca, y_train_pca)

# Making predictions
y_pred_rf = model_rfc_pca.predict(df_test_pca)

#Confusion Matrix
confusion_matrix(y_test_pca, y_pred_rf)

#Model Metrics
print('Recall Score:', recall_score(y_test_pca, y_pred_rf))
print('Precision Score:', precision_score(y_test_pca, y_pred_rf))
print('Accuracy Score:', accuracy_score(y_test_pca, y_pred_rf))

### Hyper parameter tuning for Random Forest

#Hyper parameter tuning
params_rf = {
    'max_depth': range(5,50,5),
    #'min_samples_leaf': range(5,30,5),
    #'min_samples_split': range(5,30,5),
    'criterion': ['entropy','gini']
}

model_tuned_rf = RandomForestClassifier(random_state=42)

model_tuned_rf_cv = GridSearchCV(estimator=model_tuned_rf, param_grid=params_rf,cv=folds, scoring='recall', verbose=1,return_train_score=True)
model_tuned_rf_cv.fit(df_train_pca, y_train_pca)

#Checking best parameters using  best_estimator_
model_tuned_rf_cv.best_estimator_

#Predicting on test data
y_pred_rf_tuned = model_tuned_rf_cv.predict(df_test_pca)

#confusion matrix
confusion_matrix(y_test_pca, y_pred_rf_tuned)

#Model Metrics
print('Recall Score:', recall_score(y_test_pca, y_pred_rf_tuned))
print('Precision Score:', precision_score(y_test_pca, y_pred_rf_tuned))
print('Accuracy Score:', accuracy_score(y_test_pca, y_pred_rf_tuned))

#### Comparing Random Forest business metric with and without hyper parameter tuning
Random Forest without hyper parameter tuning Recall Score: 0.8957194899817851 

Random Forest with hyper parameter tuning Recall Score: 0.947632058287796

## XG Boosting

#Model building using XG Boosting 
xgb = XGBClassifier(max_depth = 20, random_state=42)

#Fitting on train data
xgb.fit(df_train_pca, y_train_pca)

#Predicting on test data
y_pred_xgb = xgb.predict(df_test_pca)

#Confusion Matrix
confusion_matrix(y_test_pca, y_pred_xgb)

#Model Metrics
# accuracy
print("Accuracy Score:", accuracy_score(y_test_pca, y_pred_xgb))

# precision
print("Precision Score:", precision_score(y_test_pca, y_pred_xgb))

# recall/sensitivity
print("Recall Score:", recall_score(y_test_pca, y_pred_xgb))

## Hyper parameter tuning for XG Boosting

#Hyper parameter tuning for XG Boosting
params_xgb = {
    'max_depth': range(5,50,5),
    #'gamma': [0.01, 0.1,0.3,0.5, 0.6, 0.7],
    #'learning_rate': [0.01, 0.1,0.3,0.5, 0.6, 0.7]
}

model_tuned_xgb = XGBClassifier(random_state=42)

model_tuned_xgb_cv = GridSearchCV(estimator=model_tuned_xgb, param_grid=params_xgb,cv=folds, scoring='recall', verbose=1,return_train_score=True)
model_tuned_xgb_cv.fit(df_train_pca, y_train_pca)

#Checking best parameters using  best_estimator_
model_tuned_xgb_cv.best_estimator_

#Predicting on test data
y_pred_xgb_tuned = model_tuned_xgb_cv.predict(df_test_pca)

#confusion matrix
confusion_matrix(y_test_pca, y_pred_xgb_tuned)

#Model Metrics
# accuracy
print("Accuracy Score:", accuracy_score(y_test_pca, y_pred_xgb_tuned))

# precision
print("Precision Score:", precision_score(y_test_pca, y_pred_xgb_tuned))

# recall/sensitivity
print("Recall Score:", recall_score(y_test_pca, y_pred_xgb_tuned))

#### Comparing  XG Boosting business metric with and without hyper parameter tuning
XG Boosting without hyper parameter tuning Recall Score: 0.964936247723133

XG Boosting with hyper parameter tuning Recall Score: 0.9685792349726776

### ---------------------------------------------**********************---------------------------------------------------

# Business Objectives from the analysis

Logistic:

1) In the 8th month if the customer day of the last recharge date is with in 5th of that month then there is more chances of churn with the confidence of 90%

	if it is after 25th then less chances of churn with confidence of 68%
    
2) If number of times recharged in the 8th month range is 0-5 then more chances of churn with the confidence of 63%


3) If customer is fb_user in 8th month then less chances of churn with the confidence of 63%


4) If the total recharge data in 8th month is greater than 0 then less chances of churn with the confidence of 75%


5) If sachet 2g data is greater than 0 then less chances of churn with the confidence of 72%

Decision Trees:

If below particular case conditions are satisfied then corresponding churn percentage can be found


1) Case 1: With churn percentage of 96.74%

	total incoming mins of usage in 8th month <= 30.02
    
	total data used in 8th month <= 35.018
    
	total outgoing mins of usage in 8th month <= 0.22
    
	3g data volume of the 7th month <= 3.5
    
    

2) Case 2: With churn percentage of 89.56%

	total incoming mins of usage in 8th month <= 30.02
    
	total data used in 8th month <= 35.018
    
	total outgoing mins of usage in 8th month in the range of (0.22, 453.405)
    
	last recharge amount in 8th month <= 3.5
    
    
3) Case 3: With churn percentage of 84.09%

	total incoming mins of usage in 8th month > 30.02    
    
	outgoing roaming mins of usage in 8th month > 0.005   
    
	local incoming mins of usage in 8th month <= 191.36  
    
	outgoing roaming mins of usage in 8th month <= 1.005   
    
	local same network outgoing mins of usage in 6th month <= 0.39



# Model Selection

Main business objective metric is Recall which mainly deals with churn.

A reasonable number and variety of different models are attempted and can conclude that Tree based algorithm giving good results for this case study

Decision tree giving Recall score of 96.7% without any PCA and hyper parameter tuning

The best suggested model according to the recall score is XGBoost which is having almost score of 96.8%

But as per computation and business requirements even Decision Tree also does better

