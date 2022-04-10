from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#load data
fuel = pd.read_csv("./fuelData/FuelConsumptionFinal.csv", header = None)

#drop campaign related columns
# cols = ['ModelYear','Fuel Type','Fuel_Consumption(City(L/100 km)','Fuel_Consumption(Hwy(L/100 km))','Fuel_Consumption(Comb(L/100 km))','Fuel_Consumption(Comb(mpg))','Smog_Rating','CO2 Rating']
# fuel= fuel.drop(cols, axis=1)
X = fuel.iloc[:, :-1].values
y = fuel.iloc[:, -1].values

#extract numeric features 
numeric_data = fuel.iloc[:, [3,4,6,7]].values
numeric_df = pd.DataFrame(numeric_data)
numeric_df.columns = ['EngineSize', 'Cylinders','CORating','Smog_Rating']



#extract categoric features
X_categoric = fuel.iloc[:, [0,1,2,5]].values

#onehotencoding
ohe = OneHotEncoder()
categoric_data = ohe.fit_transform(X_categoric).toarray()
categoric_df = pd.DataFrame(categoric_data)
categoric_df.columns = ohe.get_feature_names()

#combine numeric and categorix
X_final = pd.concat([numeric_df, categoric_df], axis = 1)

#train model
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_final, y)

#create flask instance
app = Flask(__name__)

#create api
@app.route('/apj', methods=['GET', 'POST'])
def predict():
    #get data from request
    data = request.get_json(force=True)
    data_categoric = np.array([data["Make"], data["Model"], data["VehicleClass"], data["Transmission"]])
    data_categoric = np.reshape(data_categoric, (1, -1))
    data_categoric = ohe.transform(data_categoric).toarray()
 
    data_EngineSize = np.array([data["EngineSize"]])
    data_EngineSize = np.reshape(data_EngineSize, (1, -1))
    # data_age = np.array(age_std_scale.transform(data_age))

    data_Cylinders = np.array([data["Cylinders"]])
    data_Cylinders= np.reshape(data_Cylinders, (1, -1))
    # data_balance = np.array(balance_std_scale.transform(data_balance))

    data_CORating = np.array([data["CORating"]])
    data_CORating= np.reshape(data_CORating, (1, -1))
    # data_balance = np.array(balance_std_scale.transform(data_balance))

    data_Smog_Rating = np.array([data["Smog_Rating"]])
    data_Smog_Rating= np.reshape(data_Smog_Rating, (1, -1))

    data_final = np.column_stack((data_EngineSize, data_Cylinders, data_CORating, data_Smog_Rating))
    data_final = pd.DataFrame(data_final, dtype=object)

    #make predicon using model
    prediction = rfc.predict(data_final)
    return Response(json.dumps(prediction[0]))
