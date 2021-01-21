import pandas_datareader.data as web
import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
from dash.dependencies import Input, Output
from main import DataPrep, ModelDesign, Prediction, Forecast
import pandas as pd


def StockData(start,end,input_data,value):
	#start = start[:10]
	#start = datetime.datetime.strptime(start, "%Y-%m-%d")
	#end = end[:10]
	#end =datetime.datetime.strptime(end, "%Y-%m-%d")

	df = web.DataReader(input_data, 'yahoo',start, end)

	if (value==1):
		data = df['Open']
	elif (value==2):
		data = (df['High']+df['Low'])/2
	elif (value==3):
		data = df['Adj Close']


	#start_t=pd.to_datetime(str(start))
	#start=start.strftime("%Y-%m-%d")
	#end_t=pd.to_datetime(str(end))
	#end=end.strftime("%Y-%m-%d")
	#price = price.to_list
	dff = df.reset_index()
	time = dff['Date']
	date=[]
	for i in range(len(time)):
		t=pd.to_datetime(str(time[i]))
		date.append(t.strftime('%Y-%m-%d'))

	return df,dff,data,date#,start,end

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([

	dbc.Row(
		dbc.Col([
			html.H1(
			)
		],width={'size':12})
	),

	dbc.Row(
		dbc.Col([
			html.H1(
			)
		],width={'size':12})
	),

	dbc.Row(        
		dbc.Col(
			html.H1(
				"Stock Market Dashboard",
				className='text-center text-primary mb-4'
			),width=12)
	),

	dbc.Row(
		dbc.Col([
			html.H1(
			)
		],width={'size':12})
	),

	dbc.Row(
		dbc.Col([
			html.H1(
			)
		],width={'size':12})
	),
	
	dbc.Row(
		dbc.Col([
			html.H1(
			)
		],width={'size':12})
	),

	dbc.Row([

		dbc.Col([

				#dcc.DatePickerRange(
					#id='my-date-picker-range',
					#start_date = datetime.datetime(1980,12,12),
					#end_date= datetime.datetime.now()
			    #),

			    #dcc.Input(
			    	#id='start-train-input', 
			    	#value='', 
			    	#type='text',
			    	#placeholder='Start Train',
			    	#style={'height':'50px'}
			    #),
			        
			    #dcc.Input(
			    	#id='end-train-input', 
			    	#value='', 
			    	#type='text',
			    	#placeholder='End Train'
			    #),
				dcc.Slider(
					id='slider',
					min=0,
					max=1,
					step=0.01,
					tooltip={'always_visible': True},
					value=0.8
				),      

				#dcc.Input(
					#id='days-input', 
					#value='', 
					#type='text',
					#placeholder='# of Days',
					#style={'height':'40px', 'width':'305px'},
			    #),

				dcc.Input(
					id='LSTM-layers-input', 
					value='3', 
					type='text',
					placeholder='# of LSTM Layers',
					style={'height':'40px', 'width':'305px'},
			    ), 
				dcc.Input(
					id='neurons-input', 
					value='30', 
					type='text',
					placeholder='# of Neurons in each LSTM Layer',
					style={'height':'40px', 'width':'305px'},
				),			    
				dcc.Input(
					id='Dropout-layers-input', 
					value='0', 
					type='text',
					placeholder='# of Dropout Layers',
					style={'height':'40px', 'width':'305px'},
			    ), 
				dcc.Input(
					id='Dropout-number-input', 
					value='0', 
					type='text',
					placeholder='Amount of Dropout (in decimal)',
					style={'height':'40px', 'width':'305px'},
			    ),
				#dcc.Input(
					#id='neurons-input', 
					#value='', 
					#type='text',
					#placeholder='# of Neurons in each LSTM Layer',
					#style={'height':'40px', 'width':'305px'},
			    #), 
				dcc.Input(
					id='batch-size-input', 
					value='32', 
					type='text',
					placeholder='Batch Size',
					style={'height':'40px', 'width':'305px'},
			    ), 
				dcc.Input(
					id='epochs-input', 
					value='1', 
					type='text',
					placeholder='# of Epochs',
					style={'height':'40px', 'width':'305px'},
			    ),
				dcc.Input(
					id='days-input', 
					value='60', 
					type='text',
					placeholder='# of Previous Days to Use',
					style={'height':'40px', 'width':'305px'},
			    ),
			    dcc.Input(
			    	id='future-input', 
			    	value='0', 
			    	type='text',
			    	placeholder='# of Days to Forecast',
			    	style={'height':'40px', 'width':'305px'},
			    	#size='31'
			    ),

			    dcc.Dropdown(
					id='activation-dropdown',
					options=[
					    {'label': 'Relu', 'value': 'relu'},
					    {'label': 'Sigmoid', 'value': 'sigmoid'},
					    {'label': 'Tanh', 'value': 'tanh'}
					],
					placeholder='Type of Activation',
					value='relu'
				),
			    dcc.Dropdown(
					id='optimizer-dropdown',
					options=[
					    {'label': 'adam', 'value': 'adam'},
					    {'label': 'High/Low Average', 'value': 2},
					    {'label': 'Adjusted Close', 'value': 3}
					],
					placeholder='Type of Optimizer',
					value='adam'
				),
				dcc.Dropdown(
					id='loss-dropdown',
					options=[
					    {'label': 'Mean Squared Error', 'value': 'mse'},
					    {'label': 'High/Low Average', 'value': 2},
					    {'label': 'Adjusted Close', 'value': 3}
					],
					placeholder='Type of Loss Function',
					value='mse'
				),

				dcc.Dropdown(
					id='price-dropdown',
					options=[
					    {'label': 'Open', 'value': 1},
					    {'label': 'High/Low Average', 'value': 2},
					    {'label': 'Adjusted Close', 'value': 3}
					],
					placeholder='Type of Price',
					value=1
				),

				html.Div(
					id='model-graph', 
					children = []
			),	
		],width={'offset':3, 'size':6})
	]),
	dbc.Row(
		dbc.Col([
			html.H1(
			)
		],width={'size':12})
	),
	dbc.Row([
		dbc.Col([
			dcc.Input(
				id='stock-input', 
				value='aapl',
				placeholder='Stock Tickers', 
				type='text',
				style={'width':'610px', 'height': '40px'}
			),
		],width={'offset':3, 'size':6})	
	])	
], fluid=True)

@app.callback(
    #[
    #Output(component_id='output-graph', component_property='children')
    Output(component_id='model-graph',component_property='children')
    #]
    ,
    [#Input(component_id='my-date-picker-range', component_property='start_date')
    #,Input(component_id='my-date-picker-range', component_property='end_date')
    #,Input(component_id='start-train-input', component_property='value')
    #,Input(component_id='end-train-input', component_property='value')
    Input(component_id='slider', component_property='value')
	#,Input(component_id='days-input', component_property='value')
	,Input(component_id='LSTM-layers-input', component_property='value')
	,Input(component_id='neurons-input', component_property='value')
	,Input(component_id='Dropout-layers-input', component_property='value')
	,Input(component_id='Dropout-number-input', component_property='value')
	#,Input(component_id='neurons-input', component_property='value')
	,Input(component_id='batch-size-input', component_property='value')
	,Input(component_id='epochs-input', component_property='value')
	,Input(component_id='days-input', component_property='value')
	,Input(component_id='future-input',component_property='value')
	,Input(component_id='activation-dropdown', component_property='value')
	,Input(component_id='optimizer-dropdown', component_property='value')
	,Input(component_id='loss-dropdown', component_property='value')
	,Input(component_id='price-dropdown',component_property='value')
	,Input(component_id='stock-input', component_property='value')
    ]
)

def update_graph(slider,LSTM_layers,neurons,Dropout_layers,Dropout_number,batch_size,epochs,days,future,activation,optimizer,loss,value,input_data):

	df,dff,data,date = StockData(datetime.datetime(1970,1,1),datetime.datetime.now(),input_data,value)
	#df,dff,data,date,start,end = StockData(start,end,input_data,value)
	#start_train=start
	#end_train=end
	#figure=dcc.Graph(
		#id = 'example-graph',
		#figure = {
		    #'data': [{'x': date, 'y': data, 'type': 'line', 'name': input_data},],
		    #'layout': {'title': input_data}
		#}
	#)
	print(slider)
	start_train = 0
	end_train =int((len(date)-1)*float(slider))
	#print(data)
	#print(data.tolist)
	data = np.reshape(data,(data.shape[0]))
	data = data.tolist()
	#print(data)
	#start_train=date.index(start_train)
	#end_train=date.index(end_train)
	data,x_train,y_train,x_test,y_test,maximum,start_train,end_train,data_list = DataPrep(data,int(days),start_train,end_train,1)
	trained_model,model=ModelDesign(int(LSTM_layers),int(Dropout_layers),int(Dropout_number),int(neurons),int(batch_size),int(epochs),x_train,y_train,activation,optimizer,loss)
	test_predictions, train_preditions = Prediction(x_test,x_train,model,1)
	data=data*maximum
	data = data.tolist()

	if(len(date[end_train:])-1 !=0):
		test_predictions=np.reshape(test_predictions,(test_predictions.shape[0]))
		test_predictions=test_predictions*maximum
		test_predictions=test_predictions.tolist()
		y_test = y_test*maximum
		y_test = y_test.tolist()
	else:
		test_predictions = []
		y_test = []

	train_preditions=train_preditions*maximum
	train_preditions=np.reshape(train_preditions,(train_preditions.shape[0]))
	train_preditions=train_preditions.tolist()

	if(int(future) != 0):
		future_predictions, future_data = Forecast(int(future),model,data_list,int(days))
		future_predictions = future_predictions*maximum
		future_predictions=future_predictions.tolist()
	else:
		future_predictions=[]


	model_figure=dcc.Graph(
		id = 'example-graph',
		figure = {
		    'data': [
		    {'x': date, 'y': data, 'type': 'line', 'name': 'Actual'}
		    ,{'x': date[start_train:end_train], 'y': train_preditions, 'type': 'line', 'name': 'Training'}
		    ,{'x': date[end_train+1:], 'y': test_predictions, 'type': 'line', 'name': 'Test'}
		    ,{'x': date[end_train+1:], 'y': y_test, 'type': 'line', 'name': 'Actual Test'}
		    #,{ 'x':list(range(len(data),len(future_predictions))),'y': future_predictions, 'type': 'line', 'name': 'Forecast'}
		    ],
		    'layout': {'title': input_data}
		}
	)

	return model_figure

if __name__ == '__main__':
    app.run_server(debug=True)