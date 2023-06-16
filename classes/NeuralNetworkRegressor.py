# This Python file provides the implementation of the Neural Network-Based 
# Regression utilizing the Keras framework.

# =============================================================================
# PHASE VIII: NEURAL NETWORK REGRESSION MODEL
# =============================================================================

# Import all the required Python libraries.
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt

# Class Definition
class NeuralNetworkRegressor():
    
    # Class Constructor
    def __init__(self, dimensionality, records_num, epochs, 
                 validation_split, visual_flag=True, fold_idx=None):
        # Set the dimensionality of the input patterns.
        self.dimensionality = dimensionality
        # Set the number of training patterns.
        self.records_num = records_num
        # Set the number of training epochs.
        self.epochs = epochs
        # Set the percentage of training patterns that will be used for 
        # validatio purposes during each training epoch.
        self.validation_split = validation_split
        # Set the batch size.
        self.batch_size = int(self.records_num / 4)
        # Set the training - testing visualization flag.
        self.visual_flag = visual_flag
        # Set the current fold index.
        self.fold_idx = fold_idx
        # Construct the neural network model.
        self.create_model()
    
    # Model Constructor
    def create_model(self):
        # This function defines the architecture of the neural network-based
        # regressor.
        model = Sequential()
        # model.add(Dense(2*self.dimensionality, 
                        #input_dim=self.dimensionality, 
                        #kernel_initializer='normal', activation='sigmoid'))
        # model.add(Dense(2*self.dimensionality, 
                        # kernel_initializer='normal',
                        # activation='sigmoid'))
        model.add(Dense(self.dimensionality,
                        input_dim=self.dimensionality,
                        kernel_initializer='normal',
                        activation='linear'))
        # model.add(Dense(self.dimensionality, 
                        # kernel_initializer='normal',
                        # activation='sigmoid'))
        # model.add(Dense(int(self.dimensionality/2),
                        # kernel_initializer='normal',
                        # activation='sigmoid'))
        # model.add(Dense(int(self.dimensionality/4), 
                        # kernel_initializer='normal',
                        # activation='sigmoid'))
        model.add(Dense(1,
                        kernel_initializer='normal', activation='linear'))
        model.summary()
        # Compile the neural network model.
        model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
        # Add the constructed model to the internal variables of the class.
        self.model = model
        
    # Train Model
    def train_model(self, X, Y):
        history = self.model.fit(X, Y, epochs=self.epochs, 
                                      batch_size=self.batch_size,
                                      verbose=1, 
                                      validation_split=self.validation_split)
        
        # Visualize the training accuracy metrics.
        if self.visual_flag:
            
            # Visualize training history in terms of MSE.
            plt.plot(history.history['mse'])
            plt.plot(history.history['val_mse'])
            # Set the current title string.
            if self.fold_idx is not None:
                title_string = "Model Accuracy in terms of MSE for FOLD {}".format(self.fold_idx)
            else:
                title_string = "Model Accuracy in terms of MSE"
            plt.title(title_string)
            plt.ylabel('mse')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper right')
            plt.grid()
            plt.show()
            
            # Visualize training history in terms of MAE.
            plt.plot(history.history['mae'])
            plt.plot(history.history['val_mae'])
            # Set the current title string.
            if self.fold_idx is not None:
                title_string = "Model Accuracy in terms of MAE for FOLD {}".format(self.fold_idx)
            else:
                title_string = "Model Accuracy in terms of MAE"
            plt.title(title_string)
            plt.ylabel('mae')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper right')
            plt.grid()
            plt.show()

            # Save the current history dictionary.
            self.history = history
    
    # Test Model
    def test_model(self, X, Y):
        # Get the estimated target value.
        Yest = self.model.predict(X)
        # Get the residual values between the actual and the predicted values.
        Res = Y - Yest
        # Visualize the training accuracy metrics.
        if self.visual_flag:
            # Generate a histogram plot for the acquired residual values.
            plt.hist(Res, density=False, bins=5)  # density=False would make counts
            # Set the current title string.
            if self.fold_idx is not None:
                title_string = "Error PDF for FOLD {}".format(self.fold_idx)
            else:
                title_string = "Error PDF"
            plt.title(title_string)
            plt.ylabel('Errors Number')
            plt.xlabel('Error')
            plt.grid()
            plt.show()
            
    # Get accuracy metrics for training and testing for the current fold.
    def get_accuracy_metrics(self, Xtrain, Ytrain, Xtest, Ytest):
        # Get the training accuracy scores for the current fold.
        train_scores = self.model.evaluate(Xtrain, Ytrain, verbose=0)
        # Get the testing accuracy scores for the current fold.
        test_scores = self.model.evaluate(Xtest, Ytest, verbose=0)
        # Isolate the training mse and mae.
        train_mse, train_mae = train_scores[1], train_scores[2]
        # Isolate the testing mse and mae.
        test_mse, test_mae = test_scores[1], test_scores[2]
        return train_mse, train_mae, test_mse, test_mae