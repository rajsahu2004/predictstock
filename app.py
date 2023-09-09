import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go

end  = dt.datetime.now()
start = end - dt.timedelta(days=5500)

st.set_page_config(page_title='Predictstock', page_icon=":moneybag:", layout='wide', initial_sidebar_state='auto')
user_input = st.text_input('Enter Stock Ticker','SBI')
df = web.DataReader(user_input, 'stooq',start,end).reset_index()
st.subheader('Data from last 15 years')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig= go.Figure(data=[go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price ($)',
)
st.plotly_chart(fig, use_container_width=True)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig= go.Figure(data=[go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price')])
fig.add_trace(go.Scatter(
                x=df.Date,
                y=ma100,
                name='Moving Average 100',
                line=dict(color='blue', width=1)))
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price ($)',
)
st.plotly_chart(fig, use_container_width=True)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig= go.Figure(data=[go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price')])
fig.add_trace(go.Scatter(
                x=df.Date,
                y=ma100,
                name='Moving Average 100',
                line=dict(color='blue', width=1)))
fig.add_trace(go.Scatter(
                x=df.Date,
                y=ma200,
                name='Moving Average 200',
                line=dict(color='yellow', width=1)))
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price ($)',
)
st.plotly_chart(fig, use_container_width=True)

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_train_date = pd.DataFrame(df['Date'][0:int(len(df)*0.7)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])
data_test_date = pd.DataFrame(df['Date'][int(len(df)*0.7):int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_train_array = scaler.fit_transform(data_train)

X_train = []
y_train = []

for i in range(100, data_train_array.shape[0]):
    X_train.append(data_train_array[i-100:i])
    y_train.append(data_train_array[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

model = load_model('keras_model.h5')

past_100_days = data_train.tail(100)
final_df = pd.concat((past_100_days, data_test), axis=0)
input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

X_test, y_test = np.array(X_test), np.array(y_test)

y_predicted = model.predict(X_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

st.subheader('Predictions vs Actual')
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data_test_date['Date'], y=y_test, mode='lines', name='Actual Price'))
fig1.add_trace(go.Scatter(x=data_test_date['Date'], y=y_predicted[:,0], mode='lines', name='Predicted Price'))
fig1.update_layout(
    xaxis_title='Date',
    yaxis_title='Price ($)',
)
st.plotly_chart(fig1, use_container_width=True)



rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(y_predicted)),2)))
st.subheader('Root Mean Square Error')
st.write(rms)
percent_accuracy = []
for i in range(len(y_test)):
    percent_accuracy.append(100 - (abs(y_test[i]-y_predicted[i])/y_test[i])*100)
st.subheader('Accuracy of the model')
st.write('Median Accuracy: ', np.median(percent_accuracy))
