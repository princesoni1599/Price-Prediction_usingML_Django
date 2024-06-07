from django.shortcuts import render
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for rendering plots
import io
import numpy as np
import urllib, base64
import pandas as pd

def matplot(request):
    df = pd.read_csv(r'/Users/princesoni/Desktop/BitCoin-Price-Prediction-main/matplot/bitcoin (2).csv')

    # Initial price and Date Graph
    df.plot()
    plt.xlabel('Date from 2020-04-13 to 2021-03-12')
    plt.ylabel('Market Price')
    plt.xlim(1, 335)
    
    # Save the initial plot to a PNG image in base64 format
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    plt.clf()
    
    # Drop the 'Timestamp' column
    df.drop(['Timestamp'], axis=1, inplace=True)
    
    prediction_days = 30  # n = 30 days
    df['Prediction'] = df[['price']].shift(-prediction_days)
    X = np.array(df.drop(['Prediction'], axis=1))
    
    # Remove the last 'n' rows where 'n' is the prediction_days
    X = X[:len(df) - prediction_days]
    
    # Create another column (the target or dependent variable) shifted 'n' units up
    y = np.array(df['Prediction'])
    
    # Get all of the y values except the last 'n' rows 
    y = y[:-prediction_days]
    
    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    prediction_days_array = np.array(df.drop(['Prediction'], axis=1))[-prediction_days:]
    
    # Linear regression model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test) * 100
    
    # Predictions
    linear_prediction = linear.predict(prediction_days_array)
    
    # Plot the predictions
    plt.plot(linear_prediction)
    plt.xlabel('Date from 2021-03-13 to 2021-04-12')
    plt.ylabel('Market Price')
    
    # Save the prediction plot to a PNG image in base64 format
    fig1 = plt.gcf()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    string1 = base64.b64encode(buf1.read())
    uri1 = urllib.parse.quote(string1)
    
    # Render the template with the plots and accuracy
    return render(request, 'home.html', {'data1': uri1, 'data': uri, 'accuracy': acc})
