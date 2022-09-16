from cgitb import reset
from django.shortcuts import render, HttpResponse
from urllib import request

#Required Machine Learning Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor


# Create your views here.

#Home Page View
def index(request):
    return render(request,'index.html')

#Predict Page View
def predict(request):
    return render(request,'predict.html')

#Prediction Result
def result(request):
    df=pd.read_csv(r"C:\Users\ABC\OneDrive\Desktop\KarachiHousePrice\KarachiHousePricePrediction\Prop.csv")
    df2 = df[df["city"] == "Karachi"]
    df3 = df2.drop(["location_id","page_url","province_name","locality","area_marla","year","month","day","agency","agent","latitude","longitude","property_id","property_type","price_bin","purpose","date_added","city","area"],axis='columns')
    df3 = df3.reset_index()
    df3 = df3.drop("index",axis=1)
    df4 = df3.drop(df3[(df3['baths']==0) & (df3['bedrooms'] > 3)].index)
    df4.drop(df4[(df4['bedrooms']==0) | (df4['baths']==0)].index, inplace=True)
    df4['price_per_sqft'] = df4['price'] / df4['area_sqft']
    df4['location'] = df4['location'].apply(lambda x: x.strip())
    location_stats = df4.groupby('location')['location'].agg('count').sort_values(ascending=False)
    locations_less_than_10 = location_stats[location_stats <= 10]
    df4['location'] = df4['location'].apply(lambda x:'others' if x in locations_less_than_10 else x)
    df4[df4['area_sqft'] / df4['bedrooms'] < 300]
    df4.drop(df4[df4['area_sqft'] / df4['bedrooms'] < 300].index, inplace = True)
    
    #removing price_per_sqft outliers:
    def remove_pps_outliers(df):
     df_out = pd.DataFrame()
     for key,subdf in df.groupby('location'):
        m = np.mean(subdf['price_per_sqft'])
        std = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft'] > (m-std)) & (subdf['price_per_sqft'] <= (m+std))]
        df_out = pd.concat([df_out,reduced_df], ignore_index=True)
     return df_out
    
    df5 = remove_pps_outliers(df4)
    
    def remove_bhk_outliers(df):
     exclude_indices = np.array([])
     for location, location_df in df.groupby("location"):
        bhk_stats = {}
        for bedroom, bedroom_df in location_df.groupby("bedrooms"):
            bhk_stats[bedroom] = {
                'mean' : np.mean(bedroom_df["price_per_sqft"]),
                'std' : np.std(bedroom_df["price_per_sqft"]),
                'count': bedroom_df.shape[0]
            }
        for bedroom, bedroom_df in location_df.groupby("bedrooms"):
            stats = bhk_stats.get(bedroom - 1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bedroom_df[bedroom_df['price_per_sqft'] < (stats['mean'])].index.values)
     return df.drop(exclude_indices, axis="index")
    
    df6 = remove_bhk_outliers(df5)

    df6.groupby('location')['location'].agg('count').sort_values(ascending=False).head(40)
    #5000 price per square ft is the most common price
    df6[df6['baths'] > df6['bedrooms']]
    df6[df6['baths'] > (df6['bedrooms'] + 2)]
    df7 = df6.drop(df6[df6['baths'] > (df6['bedrooms'] + 2)].index)
    df8 = df7.drop("price_per_sqft", axis=1)
    dummies = pd.get_dummies(df8['location'])
    df9 = pd.concat([df8, dummies.drop('others', axis=1)], axis="columns")
    df9 = df9.drop("location", axis=1)
    #Then we concat them with our data.
    X = df9.drop('price', axis=1)
    y = df9['price'] 
    from sklearn.model_selection import train_test_split # for dividing data into training and test sets
    
    #Using Decision Tree As it Gives 93% Accuracy
    dtr = DecisionTreeRegressor(criterion='friedman_mse', splitter='random', random_state=0)
    dtr.fit(X, y)
    X.columns
    
    def predict_price(location, sqft, bedrooms, baths):
     loc_index = np.where(X.columns==location)[0][0]
    
     x = np.zeros(len(X.columns))
     x[0] = baths
     x[1] = sqft
     x[2] = bedrooms
     if loc_index >= 0:
         x[loc_index] = 1
     return dtr.predict([x])[0] / 100000


    var1 = request.GET['n1']
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    
    pred=predict_price(var1,var2,var3,var4)
    price = "The predicted price is Rs. " + str(pred) + " Lakhs"
    return render(request,'predict.html',{"result":price})

