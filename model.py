import pandas as pd
import numpy as np
import pickle

data=pd.read_csv("C:/Users/Siva/Desktop/Tennis/ATP Dataset_2012-01 to 2017-07_Int_V4.csv")
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Initialize the LabelEncoder
label_encoder = LabelEncoder()


# Encode the 'Player1' and 'Player2' columns
data['Player1_Encoded'] = label_encoder.fit_transform(data['Player1'])
data['Player2_Encoded'] = label_encoder.fit_transform(data['Player2'])

# Encode 'Winner' based on the relation to 'Player1' and 'Player2'
def encode_winner(row):
    if row['Winner'] == row['Player1']:
        return 1  # Player1 is the winner
    else:
        return 0  # Player2 is the winner

data['Winner_Encoded'] = data.apply(encode_winner, axis=1)


#print(data[['Player1', 'Player2', 'Winner', 'Player1_Encoded', 'Player2_Encoded', 'Winner_Encoded']].head())


df=pd.read_csv("C:/Users/Siva/Desktop/Tennis/cleane_data.csv")
df = df.drop('Date', axis=1)
#drop player1,player2 and winner column
df = df.drop(['Player1', 'Player2', 'Winner'], axis=1)
#drop 'ATP', 'Tournament_Int', 'Series_Int', 'Court_Int', 'Surface_Int','Round_Int', 'Best_of' columns
df = df.drop(['ATP', 'Tournament_Int', 'Series_Int', 'Court_Int', 'Surface_Int','Round_Int', 'Best_of'], axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop('Winner_Encoded', axis=1))

#defining x & y
x= scaled_data
y=df['Winner_Encoded']



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#importing the best library
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV

# Define the model
gnb = GaussianNB()

# Define the parameter distribution
param_dist = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=gnb, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, verbose=2)

# Fit RandomizedSearchCV
random_search.fit(x_train, y_train)

# # Print the best parameters and estimator
# print(f"Best Parameters: {random_search.best_params_}")
# print(f"Best Estimator: {random_search.best_estimator_}")

# # Evaluate the model
# y_pred = random_search.predict(x_test)
# print(classification_report(y_test, y_pred))


with open('label_encoder.pkl', 'wb') as le_file:
    label_encoder = pickle.dump(label_encoder,le_file)

with open('scaler.pkl', 'wb') as scaler_file:
    scaler = pickle.dump(scaler,scaler_file)

with open('best_model.pkl', 'wb') as model_file:
    pickle.dump(random_search, model_file)

