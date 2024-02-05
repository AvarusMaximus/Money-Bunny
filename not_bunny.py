# Simple Artificial Neural Network (ANN) model
def ann(data):
    '''
    This accepts scaled data and returns the mae, mse and r2 score of the model and the predicted and actual values
    '''
    
    # Importing the required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Input
    from sklearn.preprocessing import MinMaxScaler
    
    # Scaling the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    # Splitting the data into training and testing data
    train_data = data[:int(0.7*len(data))]
    test_data = data[int(0.7*len(data)):]
    
    X_train, y_train = [], []
    past_days = 20 # We will use the Adj Close price data of the past 20 days to predict the next day's Adj Close price
    
    for i in range(past_days, len(train_data)):
        X_train.append(train_data[i-past_days:i, 0])
        y_train.append(train_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train) # Converting the lists to numpy arrays
    
    # Buiding the Artificial Neural Network Model using Keras
    model = Sequential()
    model.add(Input(20, shape=(past_days,)))
    model.add(Dense(10, activation='relu', input_dim=past_days))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.summary())
    
    # Training the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split = 0.05, verbose = 1)
    '''
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    # The training loss indicates how well the model is fitting the training data, while the validation loss indicates how well the model fits new data
    plt.legend()
    plt.show()
    '''
    # Testing the model
    X_test, y_test = [], []
    for i in range(past_days, len(test_data)):
        X_test.append(test_data[i-past_days:i, 0])
        y_test.append(test_data[i, 0])
        
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    y_pred = model.predict(X_test)
    
    y_test = y_test.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    
    # inverse scaling the data
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    
    # Reshaping the data
    y_pred = y_pred.reshape(-1)
    y_test = y_test.reshape(-1)
    
    # Calculating the Mean Absolute Error, Mean Squared Error and R2 Score
    mae = np.mean(np.abs(y_pred - y_test))
    mse = np.mean((y_pred - y_test)**2)
    r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
    
    return mae, mse, r2, y_pred, y_test

# LSTM model with 1 layer
def lstm_1d(data):
    '''
    This accepts scaled data and returns the mae, mse and r2 score of the model and the predicted and actual values
    '''
    
    # Importing the required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, Input
    from sklearn.preprocessing import MinMaxScaler
    
    # Scaling the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    # Splitting the data into training and testing data
    train_data = data[:int(0.7*len(data))]
    test_data = data[int(0.7*len(data)):]
    
    X_train, y_train = [], []
    past_days = 20 # We will use the Adj Close price data of the past 20 days to predict the next day's Adj Close price
    
    for i in range(past_days, len(train_data)):
        X_train.append(train_data[i-past_days:i, 0])
        y_train.append(train_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train) # Converting the lists to numpy arrays
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # Reshaping the data to fit the LSTM model (Requires 3 dimensions)
    
    # Buiding the Artificial Neural Network Model using Keras
    model = Sequential()
    model.add(Input(20, shape=(past_days,)))
    model.add(LSTM(10, activation='relu', return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.summary())
    
    # Training the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split = 0.05, verbose = 1)
    '''
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    # The training loss indicates how well the model is fitting the training data, while the validation loss indicates how well the model fits new data
    plt.legend()
    plt.show()
    '''
    # Testing the model
    X_test, y_test = [], []
    for i in range(past_days, len(test_data)):
        X_test.append(test_data[i-past_days:i, 0])
        y_test.append(test_data[i, 0])
        
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    y_pred = model.predict(X_test)
    
    y_test = y_test.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    
    # inverse scaling the data
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    
    # Reshaping the data
    y_pred = y_pred.reshape(-1)
    y_test = y_test.reshape(-1)
    
    # Calculating the Mean Absolute Error, Mean Squared Error and R2 Score
    mae = np.mean(np.abs(y_pred - y_test))
    mse = np.mean((y_pred - y_test)**2)
    r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
    
    return mae, mse, r2, y_pred, y_test

# LSTM model with 2 layers
def lstm_2d(data):
    '''
    This accepts scaled data and returns the mae, mse and r2 score of the model and the predicted and actual values
    '''
    
    # Importing the required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, Input
    from sklearn.preprocessing import MinMaxScaler
    
    # Scaling the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    # Splitting the data into training and testing data
    train_data = data[:int(0.7*len(data))]
    test_data = data[int(0.7*len(data)):]
    
    X_train, y_train = [], []
    past_days = 20 # We will use the Adj Close price data of the past 20 days to predict the next day's Adj Close price
    
    for i in range(past_days, len(train_data)):
        X_train.append(train_data[i-past_days:i, 0])
        y_train.append(train_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train) # Converting the lists to numpy arrays
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # Reshaping the data to fit the LSTM model (Requires 3 dimensions)
    
    # Buiding the Artificial Neural Network Model using Keras
    model = Sequential()
    model.add(Input(20, shape=(past_days,)))
    model.add(LSTM(10, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(10, activation='relu', return_sequences=False))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.summary())
    
    # Training the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split = 0.05, verbose = 1)
    '''
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    # The training loss indicates how well the model is fitting the training data, while the validation loss indicates how well the model fits new data
    plt.legend()
    plt.show()
    '''
    # Testing the model
    X_test, y_test = [], []
    for i in range(past_days, len(test_data)):
        X_test.append(test_data[i-past_days:i, 0])
        y_test.append(test_data[i, 0])
        
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    y_pred = model.predict(X_test)
    
    y_test = y_test.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    
    # inverse scaling the data
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    
    # Reshaping the data
    y_pred = y_pred.reshape(-1)
    y_test = y_test.reshape(-1)
    
    # Calculating the Mean Absolute Error, Mean Squared Error and R2 Score
    mae = np.mean(np.abs(y_pred - y_test))
    mse = np.mean((y_pred - y_test)**2)
    r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
    
    return mae, mse, r2, y_pred, y_test

# LSTM model with 3 layers
def lstm_3d(data):
    '''
    This accepts scaled data and returns the mae, mse and r2 score of the model and the predicted and actual values
    '''
    
    # Importing the required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, Input
    from sklearn.preprocessing import MinMaxScaler
    
    # Scaling the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    # Splitting the data into training and testing data
    train_data = data[:int(0.7*len(data))]
    test_data = data[int(0.7*len(data)):]
    
    X_train, y_train = [], []
    past_days = 20 # We will use the Adj Close price data of the past 20 days to predict the next day's Adj Close price
    
    for i in range(past_days, len(train_data)):
        X_train.append(train_data[i-past_days:i, 0])
        y_train.append(train_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train) # Converting the lists to numpy arrays
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # Reshaping the data to fit the LSTM model (Requires 3 dimensions)
    
    # Buiding the Artificial Neural Network Model using Keras
    model = Sequential()
    model.add(Input(20, shape=(past_days,)))
    model.add(LSTM(10, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(10, activation='relu', return_sequences=True))
    model.add(LSTM(10, activation='relu', return_sequences=False))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.summary())
    
    # Training the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split = 0.05, verbose = 1)
    '''
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    # The training loss indicates how well the model is fitting the training data, while the validation loss indicates how well the model fits new data
    plt.legend()
    plt.show()
    '''
    # Testing the model
    X_test, y_test = [], []
    for i in range(past_days, len(test_data)):
        X_test.append(test_data[i-past_days:i, 0])
        y_test.append(test_data[i, 0])
        
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    y_pred = model.predict(X_test)
    
    y_test = y_test.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    
    # inverse scaling the data
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    
    # Reshaping the data
    y_pred = y_pred.reshape(-1)
    y_test = y_test.reshape(-1)
    
    # Calculating the Mean Absolute Error, Mean Squared Error and R2 Score
    mae = np.mean(np.abs(y_pred - y_test))
    mse = np.mean((y_pred - y_test)**2)
    r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
    
    return mae, mse, r2, y_pred, y_test