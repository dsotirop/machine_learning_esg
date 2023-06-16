# This script file provides fundamemtal computational functionality for the 
# ESG Regression Problem through the utilization of a Deep Neural Network. The
# primary objective is to derive a deep neural network-based regression model 
# that will approximate the ESG index based on a set of financial variables for
# a set of greek companies.

# =============================================================================
#                      LIST OF AVAILABLE DATA SOURCES:
# =============================================================================
# [i]:    ESG                  [TARGET REGRESSION VARIABLE]
# [ii]:   ENVIRONMENTAL        [EXPLANATORY VARIABLE]
# [iii]:  GOVERNANCE           [EXPLANATORY VARIABLE]
# [iv]:   ROE                  [EXPLANATORY VARIABLE]
# [v]:    DIVIDENDS            [EXPLANATORY VARIABLE]
# [vi]:   EMPLOYEES            [EXPLANATORY VARIABLE]
# [vii]:  BOOK TO MARKET RATIO [EXPLANATORY VARIABLE]
# [viii]: NET SALES            [EXPLANATORY VARIABLE]
# [ix]:   LEVERAGE             [EXPLANATORY VARIABLE] 
# [x]:    SOCIAL               [EXPLANATORY VARIABLE]
# [xi]:   STOCK PRICES         [EXPLANATORY VARIABLE]
# [xii]:  CAPITAL EXPENDITURES [EXPLANATORY VARIABLE]
# [xiii]: TOTAL ASSETS         [EXPLANATORY VARIABLE]
# =============================================================================

# =============================================================================
#                LIST OF AVAILABLE COMPANIES:
# =============================================================================
# [i]:    ALPHA SERVICES AND HOLDINGS
# [ii]:   EUROBANK HOLDINGS 
# [iii]:  NATIONAL BANK OF GREECE
# [iv]:   PIRAEUS FINANCIAL HOLDINGS
# [v]:    GREEK ORGANISATION OF FOOTBALL PROGNOSTICS
# [vi]:   ELLAKTOR 
# [vii]:  HELLENIC TELECOMMUNICATIONS ORGANISATION
# [viii]: HELLENIQ ENERGY HOLDINGS 
# [ix]:   MOTOR OIL
# [x]:    PUBLIC POWER
# [xi]:   BANK OF GREECE
# [xii]:  FOLLI FOLLIE
# [xiii]: HELLENIC EXCHANGES HDG.
# [xiv]:  INTRACOM HOLDING
# [xv]:   MIG HOLIDINGS
# [xvi]:  TECHNICAL OLYMPIC     
# =============================================================================

# =============================================================================
#                        TIME RANGE: [2002...2021]
# =============================================================================

# Import all required Python libraries.
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from sklearn.model_selection import KFold
from classes.NeuralNetworkRegressor import NeuralNetworkRegressor
# =============================================================================
#                          FUNCTIONS DEFINTION:
# =============================================================================
# plot_histogram generates the probability density plot of a given variable.
# =============================================================================
# Input Variables:
#                 [series]: a given data series
#                 [title_str]: the title string for the new figure.
#                 [figures_path]: the local path to the figures directory
#                 [bins_num]: the number of equally-sized bins that will split 
#                             the total range of values for the given series.
# =============================================================================
def plot_histogram(series, title_str, figures_path, bins_num):
    min_val = round(min(series))
    max_val = round(max(series))
    range_val = max_val - min_val
    bin_width = round((1 / bins_num) * range_val)
    bins_range = range(min_val, max_val+bin_width, bin_width)
    n, bins, patches = plt.hist(series, bins=bins_range, color='red',
                                alpha=0.7, rwidth=0.75)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(series.name)
    plt.ylabel('Frequency')
    plt.title(title_str)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    filename = series.name + ".png"
    figure_url = os.path.join(figures_path, filename)
    plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
    plt.show()
# =============================================================================
# geneerate_numeric_histogram generates the probability density plot for all
# numeric series in the given data frame.
# =============================================================================
# Input Variables:
#                  [data]: the given data frame
#                  [figures_path]: the local path to the figures directory
#                  [variable_names]: list of variable name strings to be used
#                                    for the construction of the respective 
#                                    title strings.
#                  [bins_num]: the number of equally-sized bins that wil split 
#                              the total range of values for any given series.
# =============================================================================
def generate_numeric_histograms(data, figures_path, variable_names, bins_num):
    os.makedirs(figures_path, exist_ok=True)
    for index, col_name in enumerate(data.select_dtypes('number').columns):
        print("Generating histogram figure for {}".format(col_name))
        col_series = data[col_name]
        title_str = "{} Probability Density:".format(variable_names[index])
        plot_histogram(col_series, title_str, figures_path, bins_num)
# =============================================================================

# =============================================================================
#                          FMAIN CODE SECTION:
# =============================================================================

# =============================================================================
# PHASE 0: DEFINE A LIST CONTAINING THE NAMES OF THE VARIOUS FINANCIAL 
#          VARIABLES AND THE CORRESPONDING SERIES
# =============================================================================
variable_names = ["year", "esg", "environmental", "governance", "roe", 
                  "dividents", "employees", "book_to_market_ratio", 
                  "net_sales", "leverage", "social", "stock_price", 
                  "capital_expenditures", "total_assets"]

series_names = ["ESG", "ENVIRONMENTAL", "GOVERNACE", "ROE", "DIVIDENTS", 
                "EMPLOYEES", "BOOK TO MARKET RATIO", "NET SALES", "LEGERAGE",
                "SOCIAL", "STOCK PRICE", "CAPITAL EXPENDITURES", 
                "TOTAL ASSETS"]
# =============================================================================
# PHASE I: LOAD ORIGINAL DATA FROM EXCEL FILES TO PANDAS DATAFRAMES
# =============================================================================

# Get the size of the current terminal window.
terminal_size = os.get_terminal_size()
# Extract the number of columns that compose the current terminal window.
columns_size = terminal_size.columns 

# Set the location of the original data sources.
data_path = "data"
# Set the local figures directory.
figures_path = "figures"
# Get a list of the available data files.
data_files = os.listdir(data_path)
# Geneate the full path for each data file.
data_full_paths = [os.path.join(data_path,data_file) 
                   for data_file in data_files] 
# Generate a list of data frames in order to store the aforementioned data 
# sources.
data_frames = [pd.read_excel(data_file) for data_file in data_full_paths]

# Take into consideration that each data frame has exactly the same structure.
# The first data series stores the various years whereas the rest of the data
# series correspond to each different company. Thus, we can extract the names
# of the various companies appearing in the dataset from any data frame object.
company_names = data_frames[0].columns[1:]

# At this point you could consider printing some of the original data frames.
# data_frames[0] ... data_frames[len(data_frames)-1]

# For each data frame print column-related information.
print("{} dataframes were loaded".format(len(data_frames)))
time.sleep(2)
for idx, df in enumerate(data_frames):
    print(columns_size*"=")
    print("Reporting information  concerning data souce: {}".format(data_files[idx]))
    print("Dataframe shape: {}".format(df.shape))
    print(columns_size*"=")
    print(df.info())
    
# =============================================================================
# PHASE II: GENARATE COMPANY-RELATED DATAFRAMES
# =============================================================================

# Loop through the various companies and accumulate the respective data series
# from each data frame. The aggregated information should be stored in a new 
# list of company specific data frames.

# Initialize the list of company specific data frames.
company_data_frames = []
# Loop through the various company names.
for company_idx in range(1, data_frames[0].shape[1]):
    # Create a list aggregating the company-specific data series from the 
    # various data frames.
    company_series = [df.iloc[:,company_idx] for df in data_frames]
    # Insert the original Name series at the begining of the list storing the
    # company-specific data series.
    company_series.insert(0, data_frames[0]["Name"])
    # Concatenate the contents of the list into a new data frame.
    company_df = pd.concat(company_series, axis=1)
    # Rename the series objects of the new data frame according to the variable
    # names defined earlier.
    company_df.columns = variable_names
    # Append the newly created data frame to the list of company-specific 
    # data frames.
    company_data_frames.append(company_df)

# At this point you could consider printing some of the company-specific data 
# frames:
# company_data_frames[0] ... company_data_frames[len(company_data_frames)-1]

# For each company-specific data frame print column-related information and 
# verify the absences null values.
print(columns_size*"=")
print("{} company-specific dataframes were loaded".
      format(len(company_data_frames)))
time.sleep(4)
for idx, df in enumerate(company_data_frames):
    print(columns_size*"=")
    print("Reporting information  concerning company: {}".format(company_names[idx]))
    print("Dataframe shape: {}".format(df.shape))
    print(columns_size*"=")
    print(df.info())
    print(columns_size*"=")
    print(df.describe())

# =============================================================================
# PHASE III: GENARATE FINAL DATA FRAME BY CONCATENATING INDIVIDUAL 
#            COMPANY-SPECIFIC DATA FRAMES AND DROPING THE YEAR VARIABLE
#            REPLACING NULL VALUES WITH COLUMN MEANS
# =============================================================================

# Generate a list in order to store the slices of the various company-related 
# data frames that are going to be concatenated within a single data framed.
# Each slice will include all original variables except the year.
slices = [df.iloc[:,1:] for df in company_data_frames]

# Form the complete data frame by concatenating the various slices and reseting
# the row numbers.
data = pd.concat(slices, axis=0, ignore_index=True)

# Report some fundamental information concerning the merged data frame.
print(columns_size*"=")
print("Data frame [data] contains {0} rows and {1} columns".
      format(data.shape[0], data.shape[1]))
print(data.info())
print(data.describe())
# Check for the existence of null values.
print(data.isnull().sum()) 
print(columns_size*"=")

# Replace null values with column means.
data = data.fillna(data.mean())
# Verify the inexistence of null values.
print(data.isnull().sum())

# =============================================================================
# PHASE IV: GENERATE THE PROBABILITY DENSITY PLOTS FOR EACH VARIABLE 
# =============================================================================

# Set the number of bins for each variable pertaining to the regression 
# problem.
bins_num = 25

# Call the function for generating the correspond probability density plots.
generate_numeric_histograms(data, figures_path, series_names, bins_num)

# =============================================================================
# PHASE V: GENERATE TWO-DIMENSIONAL PLOTS FOR SOME PAIRS OF VARIABLES
# =============================================================================

# Plot the environmental versus the governance variable. The size of each data
# point will be analogous to the esg variable.
data.plot(kind="scatter", x="environmental", y="governance", alpha=0.4,
    s=data["esg"], label="ESG", figsize=(10,8),
    c="esg", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
plt.grid()
plt.title("Environmental vs Governance")
# Save figure.
figure_url = os.path.join(figures_path, "Environmental_vs_Governance.png")
plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
plt.show()

# Plot the environmental versus the social variable. The size of each data 
# point will be analogous to the esg variable.
data.plot(kind="scatter", x="environmental", y="social", alpha=0.4,
    s=data["esg"], label="ESG", figsize=(10,8),
    c="esg", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
plt.grid()
plt.title("Environmental vs Social")
# Save figure.
figure_url = os.path.join(figures_path, "Environmental_vs_Social.png")
plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
plt.show()

# Plot the governance versus the social variable. The size of each data 
# point will be analogous to the esg variable.
data.plot(kind="scatter", x="governance", y="social", alpha=0.4,
    s=data["esg"], label="ESG", figsize=(10,8),
    c="esg", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
plt.grid()
plt.title("Governance vs Social")
# Save figure.
figure_url = os.path.join(figures_path, "Governance_vs_Social.png")
plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
plt.show()

# Plot the net salve versus the employees variable. The size of each data 
# point will be analogous to the esg variable.
data.plot(kind="scatter", x="net_sales", y="employees", alpha=0.4,
    s=data["esg"], label="ESG", figsize=(10,8),
    c="esg", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
plt.grid()
plt.title("Net Sales vs Employees")
# Save figure.
figure_url = os.path.join(figures_path, "NetSales_vs_Employees.png")
plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
plt.show()

# Plot the book to market ratio versus the stock price variable. The size of 
# each data point will be analogous to the esg variable.
data.plot(kind="scatter", x="book_to_market_ratio", y="stock_price", alpha=0.4,
    s=data["esg"], label="ESG", figsize=(10,8),
    c="esg", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
plt.grid()
plt.title("Book to Market Ratio vs Stock Price")
# Save figure.
figure_url = os.path.join(figures_path, "NetSales_vs_Employees.png")
plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
plt.show()

# =============================================================================
# PHASE VI: GENERATE THE PAIRWISE CORRELATIONS AMONGST ALL VARIABLES
# =============================================================================

# Generate the [m x m] correlation matrix where m=13 is the number of all 
# variables pertaining to the regression problem.
corr_matrix = data.corr()

# Isolate the correlation between the target regression variable "esg" and all
# the other explanatory variables. Sort the resulting correlation values in 
# descending order.
esg_correlation = corr_matrix["esg"].sort_values(ascending=False)
print(columns_size*"=")
print(esg_correlation)
print(columns_size*"=")

# Generate pairwise scatter plots for the top correlated variables that have 
# been identified previously.

# Intially populate the list of highly correlated variables.
top_correlated_variables = ["esg", "social", "environmental", "governance"]
scatter_matrix(data[top_correlated_variables], figsize=(12, 8))
# Save figure.
figure_url = os.path.join(figures_path, "TopVariablesCorrelation.png")
plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
plt.show()

# =============================================================================
# PHASE VII: TRAIN - TEST REGRESSION MODEL UTILIZING 10-FOLD CROSS VALIDATION
# =============================================================================

# Set the number of folds to be used throughout the cross-validation process.
folds_num = 10

# Initialize the cross validator object.
cross_validator = KFold(folds_num, shuffle=True, random_state=0)

# Initialize a list container for storing the neural network regressor for each
# training fold.
neural_regressors = []

# Initialize list containers for storing the training and testing mse and mae
# for each fold.
mse_train, mae_train, mse_test, mae_test = [], [], [], []

# Loop through the various folds.
for fold_idx, (train_ids, test_ids) in enumerate(cross_validator.split(data)):
    print("Executing Training / Testinng Fold: {}".format(fold_idx+1))
    # Isolate training and testing data samples.
    data_train = data.loc[train_ids]
    data_test = data.loc[test_ids]
    # Initialize the standard scaler for the training subset of data.
    scaler_train = StandardScaler()
    # Initialize the standard scaler for the testing subset of data.
    scaler_test = StandardScaler()
    # Fit the standard scaler training transform on the training data.
    scaler_train.fit(data_train)
    # Fit the standard scaler testing transform on the testing data.
    scaler_test.fit(data_test)
    # Apply the learned transformation on the training data.
    data_train = scaler_train.transform(data_train)
    # Apply the learned transformation on the testing data.
    data_test = scaler_test.transform(data_test)
    # Isolate training and testing regression features Xtrain and Xtest.
    # Take into consideration the fact that all explanatory variables are 
    # stored column-wise after the first column of the respective data frames.
    # Notice that at this point variables data_train and data_test are numpy
    # arrays.
    Xtrain = data_train[:, 1:]
    Xtest = data_test[:, 1:]
    # Isolate raining and testing target regression variables Ytrain and Ytest.
    # Take into consideration the fact that the target regression variable is
    # stored within the first column of the respective data frames.
    # Notice that variables data_train and data_test are still numpy arrays.
    # Moreover, target regression variables Ytrain and Ytest need to be reshaped into 
    # into column vectors in order to be fed within the neural network 
    # architecture.
    Ytrain = data_train[:, 0].reshape(-1,1)
    Ytest = data_test[:, 0].reshape(-1,1)
    # Get the dimensionality of the input patterns.
    dim = Xtrain.shape[1]
    # Get the number of training patterns.
    records_num = Xtrain.shape[0]
    # Set the number of training epochs.
    epochs = 300
    # Set the percentage of training patterns that will be used for validation
    # purposes during each training epoch.
    validation_split = 0.1
    # Set the visualization flag to True.
    visual_flag = True
    # Instantiate the neural regressor class for the current training-testing
    # fold.
    neural_regressor = NeuralNetworkRegressor(dim, records_num, epochs, 
                                              validation_split, visual_flag,
                                              fold_idx+1)
    # Train the previously initialized neural regressor.
    neural_regressor.train_model(Xtrain, Ytrain)
    # Test the previously initialized neural regressor.
    neural_regressor.test_model(Xtest, Ytest)
    # Get the training and testing mse and mae for the current fold.
    train_mse, train_mae, test_mse, test_mae = neural_regressor.get_accuracy_metrics(
                                                                Xtrain, 
                                                                Ytrain, 
                                                                Xtest, Ytest)
    # Append the acquired measurements to the corresponding lists.
    mse_train.append(train_mse)
    mae_train.append(train_mae)
    mse_test.append(test_mse)
    mae_test.append(test_mae)
    # Store the trained model in the designated list structure.
    neural_regressors.append(neural_regressor)
    print(columns_size*"=")

# =============================================================================
# PHASE IX: CONSTRUCT ACCURACY DATA FRAME
# =============================================================================
accuracy_data = pd.DataFrame(list(zip(mse_train,mae_train,mse_test,mae_test)),
                             columns=["MSE_TRAIN","MAE_TRAIN","MSE_TEST",
                                      "MAE_TEST"])
print(accuracy_data)
print(columns_size*"=")