# **Stock Price Forecast WebApp**
## *By Azaan Azam*

Used LSTM cells to in a Recurent Neural Network to attempt to forecast future stock prices using previous price data for any company. Built a web applicaiton using Dash for the frontend. Interacted with the Yahoo Finance API to gather all necessary data in real time.

Contanerized using Docker and hosted the website on AWS using the Elastic Beanstalk instance. Avaialable at: lstmstockpricepredictor-env.eba-qwqnhjmn.us-east-2.elasticbeanstalk.com/

Web app allows users to create their own model for up to 5 publically traded companies by allowing them to alter any of the hyper parameters associated with the model.

The file main.py has all the backend functions used to process data, build the models specified, train the model and process the outputs.

app.py includes the front end gathers user input and shows the result of their models prediction and forecast. It also includes additional detail about how the model works and what each parameter means and how it affects the output.
