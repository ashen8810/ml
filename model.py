# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle

df=pd.read_csv("Bank_Personal_Loan_Modelling.CSV")

# Remove space in Col names
df.columns = df.columns.str.replace(' ', '')

# Drop ID & Zipcode
df.drop(["ID","ZIPCode"],inplace=True,axis=1) 

# Take absolute value of experiance
df['Experience']=df['Experience'].abs()

# Convert mortgage to categorical
df.loc[df['Mortgage'] > 0, 'Mortgage'] = 1

# Change data types
df.PersonalLoan = df.PersonalLoan.astype("category")
df.SecuritiesAccount = df.SecuritiesAccount.astype("category")
df.CDAccount = df.CDAccount.astype("category")
df.Online = df.Online.astype("category")
df.CreditCard= df.CreditCard.astype("category")
df.Mortgage= df.Mortgage.astype("category")

#Splitting
x = df.drop(['PersonalLoan'],axis=1)
y = df['PersonalLoan']

#spliting the data into 80/20
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

le=LabelEncoder()
cat_cols=["SecuritiesAccount","CDAccount","Online","CreditCard","Mortgage"]
x_train[cat_cols]=x_train.loc[:,cat_cols].apply(lambda col : le.fit_transform(col))
x_test[cat_cols]=x_test.loc[:,cat_cols].apply(lambda col : le.fit_transform(col))

#fit the best model
xgbcl=XGBClassifier(learning_rate= 0.1, 
                    max_depth= 7, 
                    n_estimators=1000, 
                    subsample= 1.0,
                    use_label_encoder=False)
xgbcl.fit(x_train, y_train)
y_pred=xgbcl.predict(x_test)

confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

# Saving model to disk using pickle
pickle.dump(xgbcl, open('model.pkl','wb'))

#further use
#model = pickle.load(open('model.pkl','rb'))
