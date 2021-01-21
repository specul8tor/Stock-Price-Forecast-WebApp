
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import csv
import numpy as np
import time
import datetime
import pandas_datareader.data as web

actual = [122.7,123,123,122,123.75,123.38,121.78,123.24,122.42,121.78,127.87,127.81,128.69,126.66,128.23,131.88,130.96,131.97,136.69,134.87,133.72,132.69,129.41,131.01,126.59,130.92,132.05,126.6,130.92,132.05,128.98,128.8,130.89,128.91]

def StockData(stock,price):

	raw = []
	data=[]
	date = []
	z=0
	with open(stock+'.csv','r') as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		for line in reader:
			raw.append(line)
			date.append(raw[z][0])
			if (raw[z][5] != 'null'):
				if(price==6):
					data.append((float(raw[z][2])+float(raw[z][3]))/2)
				else:
					data.append(float(raw[z][price]))
			else:	
				date.pop()
			z=z+1
	print(data)
	return date,data

def DataPrep(data,days,start_train,end_train,choice):
	print(end_train)
	maximum = max(data)
	for i in range(len(data)):
		data[i] = data[i]/maximum

	train = data[start_train:end_train]
	test_data = data[end_train+1:]
	test = data[len(data)-len(test_data)-days:]

	training = []
	label = []
	for i in range(days,len(train)):
		training.append(train[i-days:i])
		label.append(train[i])

	testing=[]
	test_labels=[]
	for i in range(days,len(test)):
		testing.append(test[i-days:i])
		test_labels.append(test[i])

	#print(data)
	data_list = data
	data = np.array(data)
	#print(training)
	x_train = np.array(training)
	#print(x_train)
	#print(x_train.shape)
	x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
	y_train = np.array(label)
	'''
	if(choice == 1):
		#print(testing)
		#print(len(testing))
		x_test = np.array(testing)
		#print(x_test)
		#print(x_test.shape)
		x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
		y_test = np.array(test_labels)
	'''
	if(len(testing) != 0):
		x_test = np.array(testing)
		x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
		y_test = np.array(test_labels)
	else:
		x_test = np.array([])
		y_test = []

	return data,x_train,y_train,x_test,y_test,maximum,start_train,end_train,data_list

def ModelDesign(LSTM_layers,Dropout_layers,Dropout_number,neurons,
	batch_size,epochs,x_train,y_train,activation,optimizer,loss,):

	model = Sequential()
	for i in range(LSTM_layers-1):
		model.add(LSTM(neurons,input_shape=(x_train.shape[1],1),return_sequences=True,activation=activation))
		if (Dropout_layers != 0):
			model.add(Dropout(Dropout_number))
			Dropout_layers = Dropout_layers -1

	model.add(LSTM(neurons,input_shape=(x_train.shape[1],1)))
	model.add(Dense(1))

	model.summary()

	model.compile(optimizer = optimizer, loss=loss)
	trained=model.fit(x_train,y_train,batch_size=batch_size,epochs= epochs)

	return trained,model

def Prediction(x_test,x_train,model,choice):

	if(x_test.shape[0] != 0):
		test_predictions = model.predict(x_test)
	else:
		test_predictions = []

	train_predictions = model.predict(x_train)

	return test_predictions,train_predictions

def Forecast(future,model,data_list,days):

	future_data = data_list
	future_predictions =[]

	for i in range(future):
		array=np.asarray(future_data[len(future_data)-days:]).astype('float32')
		array=np.reshape(array, (1,array.shape[0]))
		array=np.reshape(array, (array.shape[0],array.shape[1],1))
		future_data.append(model.predict(array))
		future_predictions.append(model.predict(array))

	future_predictions=np.array(future_predictions)
	future_predictions=np.reshape(future_predictions,(future_predictions.shape[0]))
	future_data=np.array(future_data)

	return future_predictions,future_data

def Plots(date,data,test_predictions,train_preditions,y_test,maximum,trained_model,days,end_train,future,future_predictions,choice):

	train_preditions = train_preditions*maximum
	data = data*maximum

	if (choice==1):
		test_predictions = test_predictions*maximum
		y_test = y_test*maximum
		plt.figure(figsize=(15,25))
		plt.plot(date,data,color='green',label='Stock Data')
		plt.plot(date[days:end_train],train_preditions,color='blue',label='Training Data')
		plt.plot(date[end_train+1:],test_predictions,color='red',label='Testing Data')
		plt.figure(figsize=(10,20))
		plt.plot(date[end_train+1:],y_test,color='green',label='Stock Data')
		plt.plot(date[end_train+1:],test_predictions,color='red',label='Prediction')
		'''
		ax=plt.gca()
		plt.xticks(rotation=30)
		interval=len(date)/25
		for index, label in enumerate(ax.xaxis.get_ticklabels()):
	    	if ((index % interval) != 0):
	    		label.set_visible(False)
	    plt.figure(figsize=(10,20))
		plt.plot(date[end_train+1:],y_test,color='green',label='Stock Data')
		plt.plot(date[end_train+1:],test_predictions,color='red',label='Prediction')
		ax=plt.gca()
		plt.xticks(rotation=30)
		interval=len(date[end_train+1:])/20
		for index, label in enumerate(ax.xaxis.get_ticklabels()):
		    if (index % interval != 0):
		        label.set_visible(False)'''

	if (choice==2):
		future_predictions = future_predictions*maximum
		plt.figure()
		plt.plot(future_predictions,color='black')
		plt.plot(actual,color='red')
		plt.show()
		plt.figure(figsize=(10,15))
		plt.plot(date[days:end_train],train_preditions,color='blue',label='Training Data')
		plt.plot(date,data,color='green',label='Stock Data')
		plt.plot(range(end_train,end_train+len(actual)),actual,color='red',label = 'future actual')
		plt.show()

	plt.figure()
	plt.plot(trained_model.history['loss'],color='blue')

	plt.show()

def Main():
	#Init()
	choice=int(input("1 for train/test split, 2 for future prediciton: "))
	stock=input("Stock: ")
	price=int(input("Data Type: 1 for Open, 2 for High, 3 for Low, 4 for Close, 5 for Adjusted Close, 6 for Mid Price: "))
	date, data = StockData(stock,price)

	days=int(input("Days for sequence: "))
	start_train=date.index(input("Start Training (in YYYY-MM-DD format): "))
	end_train=date.index(input("End Training (in YYYY-MM-DD format): "))
	data, x_train, y_train, x_test, y_test, maximum, start_train, end_train, data_list = DataPrep(data,days,start_train,end_train,choice)

	LSTM_layers=int(input("Number of LSTM layers: "))
	neurons=int(input("Number of neurons per LSTM layer: "))
	Dropout_layers=int(input("Number of Dropout layers: "))
	Dropout_number=float(input("Dropout percentage (in decimal): "))
	batch_size=int(input("Batch Size: "))
	epochs=int(input("Epochs: "))
	activation=input("Activation to use for LSTM: ")
	optimizer=input("Optimizer to use: ")
	loss=input("Loss Function to use: ")
	trained_model,model=ModelDesign(LSTM_layers,Dropout_layers,Dropout_number,neurons,batch_size,epochs,x_train,y_train,activation,optimizer,loss)

	test_predictions, train_preditions = Prediction(x_test,x_train,model,choice)

	if(choice==2):
		future = int(input("Number of days to Forecast: "))
		future_predictions, future_data = Forecast(future,model,data_list,days)
	else:
		future = 0
		future_predictions = 0
	Plots(date,data,test_predictions,train_preditions,y_test,maximum,trained_model,days,end_train,future,future_predictions,choice)

#Main()
def DataFrame(stock,startyear,startmonth,startday,endyear,endmonth,endday,price):

	df = web.DataReader(stock, 'yahoo', datetime.datetime(startyear,startmonth,startday), datetime.datetime(endyear,endmonth,endday))
	print(df)
	if price == 1:
		data = df.Close
	if price == 2:
		data = (df.High + df.Low)/2
	maximum = np.max(data)
	data_norm = data/maximum

#Main()
#StockData('aapl',6)
#DataFrame('aapl',1980,12,12,2021,1,15,2)
#print(web.DataReader('aapl','yahoo',datetime.datetime(2020,1,12),datetime.datetime(2021,1,6)))
