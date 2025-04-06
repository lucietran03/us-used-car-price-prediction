# pip install scikit-learn # to install scikit-learn
# pip install pandas # to install pandas
# pip install numpy # to install numpy
# pip install matplotlib # to install matplotlib
# pip install joblib # to install joblib

# Import the required libraries
import pandas as pd  # Pandas library for tabular data processing
import numpy as np  # Numpy library for arithmetic operations
import matplotlib.pyplot as plt  # Matplotlib library for plotting
from sklearn.base import BaseEstimator, TransformerMixin  # Base for creating custom estimators and transformers
from sklearn.pipeline import FeatureUnion  # Combine multiple transformers into a feature union
from sklearn.pipeline import Pipeline  # Create and manage a series of processing steps
from sklearn.preprocessing import StandardScaler  # Transform data to have a mean of 0 and a standard deviation of 1
from sklearn.impute import SimpleImputer  # Replace missing values ​​with a simple method
from sklearn.preprocessing import OneHotEncoder  # Transform categorical variables into binary variables
from sklearn.model_selection import KFold  # Split data into training and test sets for cross-validation
from statistics import mean  # Calculate mean
from sklearn.model_selection import train_test_split
import joblib  # Save and load trained models
from pandas.plotting import scatter_matrix  # Create scatter matrix plots to visualize relationships between features
# Enable inline plotting for Jupyter notebooks
import graphviz  # For visualizing decision trees or graph-based models
import seaborn as sns  # For statistical data visualization
from scipy.stats import skew  # To calculate skewness in data (asymmetry of data distribution)
import warnings  # For managing warnings in the code
import pandas as pd  # Pandas library for tabular data processing



# Read data from CSV file
file_path = r'dataset/used_cars_100000.csv' 
import_data = pd.read_csv(file_path)




print('\n> SOME FIRST DATA EXAMPLES')
import_data.head(3)

print('\n____________ DATASET INFO ____________')
print(import_data.info())       

print('\n___ COUNTS ON A FEATURE ___')
print(import_data['make_name'].value_counts()) 

print('\n > STATISTICS OF NUMERIC FEATURES (Data identified by the system has not been processed)')
import_data.describe()



# Danh sách các cột Boolean cần chuyển đổi
bool_cols = ['isCab', 'is_oemcpo', 'is_cpo', 'is_new', 'vehicle_damage_category']
# Thay thế NaN bằng False trước khi chuyển đổi
import_data[bool_cols] = import_data[bool_cols].fillna(False).astype(int)

import_data['owner_count'] = import_data['owner_count'].astype('category')  # Convert 'owner_count' column to category type
import_data['engine_displacement'] = import_data['engine_displacement'].astype('category')  # Convert 'engine_displacement' column to category type
import_data['maximum_seating'] = import_data['maximum_seating'].astype('category')  # Convert 'maximum_seating' column to category type
import_data['year'] = import_data['year'].astype('category')  # Convert 'year' column to category type
import_data['bed_length'] = import_data['bed_length'].astype('category')  # Convert 'bed_length' column to category type

# Define a function to extract numeric values and remove units
def convert_to_numeric(value):
    try:
        # Remove the unit (e.g., 'in') and convert to float
        return float(value.split()[0])
    except:
        return None  # Handle cases where conversion might fail

# Apply the function to the specified columns
import_data['bed_length'] = import_data['bed_length'].apply(convert_to_numeric)
import_data['wheelbase'] = import_data['wheelbase'].apply(convert_to_numeric)
import_data['back_legroom'] = import_data['back_legroom'].apply(convert_to_numeric)
import_data['front_legroom'] = import_data['front_legroom'].apply(convert_to_numeric)
import_data['width'] = import_data['width'].apply(convert_to_numeric)
import_data['height'] = import_data['height'].apply(convert_to_numeric)
import_data['length'] = import_data['length'].apply(convert_to_numeric)
import_data['fuel_tank_volume'] = import_data['fuel_tank_volume'].apply(convert_to_numeric)




# Function to split torque data into torque value and RPM
def split_torque(torque_str):
    try:
        torque_value = float(torque_str.split(' ')[0])
        rpm_value = int(torque_str.split('@')[1].strip().split(' ')[0].replace(',', ''))
        return pd.Series([torque_value, rpm_value])
    except (AttributeError, IndexError, ValueError):
        return pd.Series([np.nan, np.nan])

# Apply the split_torque function to the torque column
import_data[['torque_value', 'torque_rpm']] = import_data['torque'].apply(lambda x: split_torque(x) if isinstance(x, str) else pd.Series([np.nan, np.nan]))
# Combine using product and replace the original torque column
import_data['torque'] = import_data['torque_value'] * import_data['torque_rpm']
# Drop the temporary torque_value and torque_rpm columns
import_data.drop(columns=['torque_value', 'torque_rpm'], inplace=True)




# Remove unused features
data_visualization = import_data.drop(columns=['vin', 'description', 'listing_id', 'main_picture_url', 'sp_id', 'sp_name', 'trimId'])
# Create histograms for each feature with 40 bins and a figure size of 20x15 inches.
data_visualization.hist(bins=40, figsize=(20, 15))
# Uncomment to save the histogram as a PNG file with 300 dpi resolution.
# plt.savefig('Figures/hist_import_data.png', format='png', dpi=300)
# Show the histogram plots.
plt.show()



# Create a scatter matrix for the 'price' feature (only one feature, so the matrix will just be a histogram)
features_to_plot = ["price"]
scatter_matrix(import_data[features_to_plot], figsize=(12, 8))  # Note: histograms are on the main diagonal

# Uncomment to set axis limits (if needed)
# plt.axis([0, 1000000, 0, 120000])

# Show the scatter matrix.
plt.show()


# Create box plots for each feature with individual subplots
data_visualization.plot(kind='box', subplots=True, layout=(6, 6), figsize=(15, 15), sharex=False, sharey=False)

# Adjust spacing between subplots for better visualization
plt.tight_layout()

# Uncomment to save the box plots as a PNG file with 300 dpi resolution.
# plt.savefig('Figures/box_import_data.png', format='png', dpi=300)

# Show the box plots.
plt.show()



# Create a box plot for the 'price' feature
plt.figure(figsize=(10, 6))
plt.boxplot(import_data['horsepower'].dropna(), notch=False, vert=True)
plt.title('Box Plot of HorsePower')  # Add a title to the plot

# Uncomment to save the box plot as a PNG file with 300 dpi resolution.
# plt.savefig('Figures/box_car_HorsePower.png', format='png', dpi=300)

# Show the box plot.
plt.show()



# Define the important features to plot
features_to_plot = ["price", "horsepower", "mileage", "city_fuel_economy", "engine_displacement"]

# Create a scatter matrix for the specified features
scatter_matrix(import_data[features_to_plot], figsize=(24, 16), diagonal='hist')

# Adjust spacing between cells in the scatter matrix
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# Uncomment to save the scatter matrix as a PNG file with 300 dpi resolution.
# plt.savefig('Figures/scatter_mat_all_feat.png', format='png', dpi=300)

# Show the scatter matrix.
plt.show()



# Create a scatter plot with 'price' on the x-axis and 'savings_amount' on the y-axis
import_data.plot(kind="scatter", y="savings_amount", x="price", alpha=0.2)

# Uncomment to save the scatter plot as a PNG file with 300 dpi resolution.
# plt.savefig('figures/scatter_1_feat.png', format='png', dpi=300)

# Show the scatter plot.
plt.show()



# Calculate the correlation matrix for the numeric columns in the DataFrame
# This matrix shows the correlation coefficients between each pair of numeric features
corr_matrix = data_visualization.corr(numeric_only=True)
# Print the correlation matrix to view the relationships between features
corr_matrix
corr_matrix["price"].sort_values(ascending=False) # print correlation b/w a feature and other features



# Calculate the ratio of city fuel economy to highway fuel economy
data_visualization["fuel_efficiency_ratio"] = data_visualization["city_fuel_economy"] / data_visualization["highway_fuel_economy"]
# Calculate the ratio of engine power to weight (mileage used as a proxy for weight)
data_visualization["engine_power_to_weight_ratio"] = data_visualization["horsepower"] / data_visualization["mileage"]
# Calculate the correlation matrix for numeric features
corr_matrix = data_visualization.corr(numeric_only=True)
# Print the correlation of each feature with the price, sorted in descending order
print(corr_matrix["price"].sort_values(ascending=False))




# Remove the created columns after analyzing their impact
data_visualization.drop(columns=["fuel_efficiency_ratio", "engine_power_to_weight_ratio"], inplace=True)
# Remove unused features
import_data = import_data.drop(columns=['vin', 'description', 'listing_id', 'main_picture_url', 'sp_id', 'sp_name', 'trimId'])
# List of categorical variables
categorical_feat_names = ["major_options", "make_name", "model_name", "body_type", "year", "has_accidents", "fleet", "franchise_dealer", "franchise_make", "frame_damaged", "owner_count", "is_cpo", "bed_length", "is_oemcpo", "isCab", "salvage", "theft_title", "is_new", "engine_cylinders", "engine_displacement", "engine_type", "power", "torque", "city", "dealer_zip", "cabin", "exterior_color", "listing_color", "interior_color", "fuel_type", "savings_amount", "transmission", "transmission_display", "trim_name", "vehicle_damage_category", "wheel_system", "wheel_system_display"]
# List of numerical variables
numeric_feat_names = ["mileage", "city_fuel_economy", "highway_fuel_economy", "fuel_tank_volume", "horsepower", "price", "daysonmarket", "length", "width", "height", "front_legroom", "wheelbase", "back_legroom", "latitude", "longitude", "seller_rating"]
process_data = import_data.copy() # Minimize impact on original data


# Apply Log Transformation to columns in process_data:
process_data['city_fuel_economy_log'] = np.log1p(process_data['city_fuel_economy'])
process_data['daysonmarket_log'] = np.log1p(process_data['daysonmarket'])
process_data['fuel_tank_volume_log'] = np.log1p(process_data['fuel_tank_volume'])
process_data['highway_fuel_economy_log'] = np.log1p(process_data['highway_fuel_economy'])
process_data['price_log'] = np.log1p(process_data['price'])
process_data['front_legroom_log'] = np.log1p(process_data['front_legroom'])
process_data['mileage_log'] = np.log1p(process_data['mileage'])

# Apply Square Root Transformation to columns in process_data:
process_data['horsepower_sqrt'] = np.sqrt(process_data['horsepower'])

# Create a boolean feature indicating zero values
process_data['savings_amount_bool'] = process_data['savings_amount'] > 0

# Convert the new boolean feature to a boolean type (if not already boolean)
process_data['savings_amount_bool'] = process_data['savings_amount_bool'].astype(bool)

# Convert boolean to string (for pipeline)
process_data['franchise_dealer'] = process_data['franchise_dealer'].astype(str)  
process_data['savings_amount'] = process_data['savings_amount_bool'].astype(str)  

# Convert all bool columns to str
process_data[process_data.select_dtypes(['bool']).columns] = process_data.select_dtypes(['bool']).astype(str)



# Get all columns from process_data
all_columns = process_data.columns

# Filter out the categorical columns
filtered_columns = [col for col in all_columns if col not in categorical_feat_names]

# Create histograms for each feature with 50 bins and a figure size of 20x15 inches.
process_data[filtered_columns].hist(bins=50, figsize=(20, 15))

# Uncomment to save the histogram as a PNG file with 300 dpi resolution.
# plt.savefig('Figures/hist_log_import_data.png', format='png', dpi=300)

# Show the histogram plots.
plt.show()



# Drop the transformed columns
columns_to_remove = [ 'city_fuel_economy_log', 'daysonmarket_log', 'fuel_tank_volume_log', 'highway_fuel_economy_log', 'price_log', 'front_legroom_log', 'mileage_log', 'horsepower_sqrt']

# Remove the columns from process_data
process_data = process_data.drop(columns=columns_to_remove)

# Drop the boolean feature column as it's now integrated into 'savings_amount'
process_data = process_data.drop(columns=['savings_amount_bool'])



# Function to clean data by removing outliers based on specified thresholds
def manualCleaned(df, thresholds):
    processed_data = df.copy()

    for feature, thres in thresholds.items():
        if feature in processed_data.columns:
            initial_rows = processed_data.shape[0]  # Record initial row count
            lower_thres, upper_thres = thres
            mask = (processed_data[feature] >= lower_thres) & (processed_data[feature] <= upper_thres)
            processed_data = processed_data[mask]
            final_rows = processed_data.shape[0]  # Record final row count

            # Print the impact of filtering on the number of rows
            row_loss = initial_rows - final_rows
            percentage_loss = (row_loss / initial_rows) * 100
            print(f"Filtering '{feature}': {initial_rows} -> {final_rows} rows remaining ({row_loss} rows removed, {percentage_loss:.2f}% loss).")
    
    return processed_data

# Define thresholds for each feature to filter out outliers
thresholds = {
    'back_legroom': (35, 44),
    'city_fuel_economy': (10, 30),
    'daysonmarket': (0, 292),
    'front_legroom': (38, 45),
    'fuel_tank_volume': (10, 25),
    'height': (60, 80),
    'highway_fuel_economy': (10, 40),
    'horsepower': (0, 450),
    'length': (160, 210),
    'longitude': (-119, -60),
    'mileage': (0, 60000),
    'price': (0, 80000),
    'seller_rating': (3.1, 5),
    'wheelbase': (85, 130),
    'width': (70, 90)
}

# Apply the outlier removal process to the entire dataset (not just numeric features)
processed_data = manualCleaned(process_data, thresholds)



# Split the dataset into main and testing datasets (80% train, 20% test)
train_set, test_set = train_test_split(processed_data, test_size=0.2, random_state=42, shuffle=True)

print('__Training and Testing datasets__')
print(f'{len(train_set)} training examples')  # Print the number of examples in the training set
print(f'{len(test_set)} test examples')        # Print the number of examples in the test set


train_set_labels = train_set["price"].copy()
train_set = train_set.drop(columns = "price")
test_set_labels = test_set["price"].copy()
test_set = test_set.drop(columns = "price")



# Update column list after removing 'price' column
numeric_feat_names = [col for col in numeric_feat_names if col != 'price']
categorial_feat_names = [col for col in categorical_feat_names if col != 'price']



# Transformer to select specific columns from a dataframe
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        # Initialize with the list of feature (column) names to be selected
        self.feature_names = feature_names
    
    def fit(self, dataframe, labels=None):
        # fit method doesn't need to do anything for this transformer
        return self
    
    def transform(self, dataframe):
        # Select the columns specified in feature_names and return their values as a numpy array
        return dataframe[self.feature_names].values  
    
class MyFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_power_to_displacement=True, add_mileage_per_day=True, add_average_fuel_economy=True):
        self.add_power_to_displacement = add_power_to_displacement
        self.add_mileage_per_day = add_mileage_per_day
        self.add_average_fuel_economy = add_average_fuel_economy

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        
        # Add power to engine displacement ratio feature
        if self.add_power_to_displacement:
            horsepower_id, engine_displacement_id = 4, 2  # Adjust these indices as per your data
            if horsepower_id < X.shape[1] and engine_displacement_id < X.shape[1]:
                power_to_displacement_ratio = X[:, horsepower_id] / X[:, engine_displacement_id]
                X = np.c_[X, power_to_displacement_ratio]
            else:
                raise IndexError(f"Invalid column index for power-to-displacement: {horsepower_id} or {engine_displacement_id}")

        # Add mileage per day on the market feature
        if self.add_mileage_per_day:
            mileage_id, daysonmarket_id = 7, 1  # Adjust these indices as per your data
            if mileage_id < X.shape[1] and daysonmarket_id < X.shape[1]:
                mileage_per_day = X[:, mileage_id] / (X[:, daysonmarket_id] + 1)
                X = np.c_[X, mileage_per_day]
            else:
                raise IndexError(f"Invalid column index for mileage per day: {mileage_id} or {daysonmarket_id}")

        # Add average fuel economy feature
        if self.add_average_fuel_economy:
            city_fuel_economy_id, highway_fuel_economy_id = 0, 3  # Adjust these indices as per your data
            if city_fuel_economy_id < X.shape[1] and highway_fuel_economy_id < X.shape[1]:
                average_fuel_economy = (X[:, city_fuel_economy_id] + X[:, highway_fuel_economy_id]) / 2
                X = np.c_[X, average_fuel_economy]
            else:
                raise IndexError(f"Invalid column index for fuel economy: {city_fuel_economy_id} or {highway_fuel_economy_id}")

        return X




# Categorical pipeline for processing categorical features
categorical_pipeline = Pipeline([
    ('selector', ColumnSelector(categorial_feat_names)), # Selects only the categorical columns specified
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="NO INFO", copy=True)),
    # Impute missing values in the categorical columns by filling with "NO INFO"
    ('cat_encoder', OneHotEncoder(sparse_output=True, handle_unknown='ignore'))  
    # One-hot encode the categorical data, outputting a sparse matrix to save memory
    # 'handle_unknown=ignore' ensures that unseen categories in the test data are ignored (won't throw errors)
])

# Pipeline for numerical features
numeric_pipeline = Pipeline([
    ('selector', ColumnSelector(numeric_feat_names)), # Select the numeric columns based on the provided feature names
    
    # Handle missing values by replacing them with the median of each column
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)),  # copy=False: imputation will be done in-place 
    
    # Add custom features using MyFeatureAdder (based on car attributes)
    ('attribs_adder', MyFeatureAdder(add_power_to_displacement=True, add_mileage_per_day=True, add_average_fuel_economy=True)),
    
    # Standardize features by removing the mean and scaling to unit variance
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True))  # Scale features to zero mean and unit variance
])



full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", numeric_pipeline),
    ("cat_pipeline", categorical_pipeline) ])



# Convert categorical columns to strings
def preprocess_categorical(df, categorical_features):
    for col in categorical_features:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int).astype(str)  # Convert booleans to strings
        else:
            df[col] = df[col].astype(str)  # Ensure all categorical data is of string type
    return df
# Assume train_set is your DataFrame
train_set = preprocess_categorical(train_set, categorial_feat_names)

# Fit the full pipeline on the training dataset and transform the data
processed_train_set_val = full_pipeline.fit_transform(train_set)

# In thử dữ liệu đã được xử lý
print('\n____________ Processed feature values ____________')
print(processed_train_set_val[:3, :])  # Hiển thị 3 hàng đầu tiên
print(processed_train_set_val.shape)


# Print the shape of the processed data to check the number of rows and columns
print(processed_train_set_val.shape)

# Print number of numeric features, added features and columns for the one-hot vectors from the categorical data
# Calculate the actual number of one-hot encoded columns
num_numeric_features = len(numeric_feat_names)
num_added_features = 3
num_onehot_columns = processed_train_set_val.shape[1] - num_numeric_features - num_added_features

print(f'We have {num_numeric_features} numeric features + {num_added_features} added features + {num_onehot_columns} columns for one-hot encoded categorical features.')

# Save the full pipeline to a file so it can be reused later for predictions or further training
# joblib.dump(full_pipeline, r'Models/full_pipeline.pkl')



# Extract the list of one-hot encoded columns from the encoder
onehot_cols = []
for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
    onehot_cols += val_list.tolist()

# Create a list of column names for the DataFrame
# Add original column names from the training set and the new one-hot encoded columns
columns_header = train_set.columns.tolist() + onehot_cols

# Check the number of columns in the processed data
num_columns_data = processed_train_set_val.shape[1]
# Check the number of column names in the columns_header list
num_columns_header = len(columns_header)

# Compare and adjust if necessary
if num_columns_data != num_columns_header:
    # Adjust columns_header to match the number of columns in the processed data
    columns_header = columns_header[:num_columns_data]

# Create a DataFrame from the processed data using the column names
processed_train_set = pd.DataFrame(processed_train_set_val.toarray(), columns=columns_header)

# Print information about the DataFrame to check its structure
print('\n____________ Processed dataframe ____________')
print(processed_train_set.info())
print(processed_train_set.head())

# Replace invalid characters (e.g., spaces) in column names with underscores
processed_train_set.columns = processed_train_set.columns.str.replace(' ', '_').str.replace('[^\w]', '_', regex=True)



# Importing various regression models from scikit-learn
from sklearn.linear_model import LinearRegression  # Standard linear regression model
from sklearn.linear_model import ElasticNet        # Linear regression with both L1 (Lasso) and L2 (Ridge) regularization
from sklearn.linear_model import Ridge            # Linear regression with L2 regularization
from sklearn.linear_model import Lasso            # Linear regression with L1 regularization
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz    # Decision tree model for regression tasks
from sklearn.ensemble import RandomForestRegressor  # Random forest model for regression tasks
from sklearn.metrics import mean_squared_error    # Function to compute the mean squared error (MSE)
from sklearn.metrics import r2_score              # Function to compute the R-squared score
from sklearn.ensemble import AdaBoostRegressor    # AdaBoost regressor
from sklearn.ensemble import GradientBoostingRegressor  # Gradient boosting regressor



def r2_score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels) # Calculate the R-squared score for the model on the training data
    prediction = model.predict(train_data) # Generate predictions using the trained model on the training data
    mse = mean_squared_error(labels, prediction) # Calculate the Mean Squared Error (MSE) between the actual and predicted labels
    rmse = np.sqrt(mse) # Compute the Root Mean Square Error (RMSE) from the MSE
    return r2score, rmse # Return the R-squared score and RMSE



def store_model(model, model_name=""):
    # NOTE: joblib is faster than Python's pickle for saving models
    # INFO: Each file can only store one object
    if model_name == "":
        # Use the model's class name if no name is provided
        model_name = type(model).__name__
    joblib.dump(model, 'Models/' + model_name + '_model.pkl')

def load_model(model_name):
    # Load the model from a file
    model = joblib.load('Models/' + model_name + '_model.pkl')
    return model



model = LinearRegression() # Create a LinearRegression model
model.fit(processed_train_set_val, train_set_labels) # Train the model on the training data

# Print the learned parameters of the model
print('\n____________ LinearRegression ____________')
print('Learned parameters: ', model.coef_, model.intercept_)

# Call the function with the model, training data, and labels, and get R2 score and RMSE
r2score, rmse = r2_score_and_rmse(model, processed_train_set_val, train_set_labels)

# Print the R-squared score and RMSE
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

# Predict the prices
predictions = model.predict(processed_train_set_val)

# Select the first column to visualize
x = processed_train_set_val[:, 0].toarray().flatten()  # Convert CSR matrix to 1D array

# Plot scatter of the actual data
plt.scatter(x, train_set_labels, color='black', alpha=0.6, label="Actual Data")

# Plot scatter of predictions instead of connecting them with lines
plt.scatter(x, predictions, color='blue', alpha=0.6, label="Predicted Data")

# Adding labels and legend
plt.xlabel('Feature (First column of processed_train_set_val)')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot predicted prices vs actual prices
plt.scatter(train_set_labels, predictions, color='blue', alpha=0.6)
plt.plot([min(train_set_labels), max(train_set_labels)], [min(train_set_labels), max(train_set_labels)], color='red', lw=2)  # Line y=x
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Prices')
plt.show()

# Calculate residuals
residuals = train_set_labels - predictions

# Plot residuals
plt.scatter(predictions, residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='red', lw=2)  # Zero error line
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals vs Predicted Prices')
plt.show()

print("\nInput data: \n", train_set.iloc[0:9]) # Print the first 9 rows of the input training data

# Print the predictions of the model for the first 9 samples, rounded to 1 decimal place
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))

print("Labels:      ", list(train_set_labels[0:9])) # Print the true labels for the first 9 samples

# Save the specified model to a file using the store_model function
# Uncomment the following line to save the model to a file
# store_model(model)




model = Ridge() # Create a RidgeRegression model
model.fit(processed_train_set_val, train_set_labels) # Train the model on the training data

# Print the learned parameters of the model
print('\n____________ RidgeRegression ____________')
print('Learned parameters: ', model.coef_, model.intercept_)

# Call the function with the model, training data, and labels, and get R2 score and RMSE
r2score, rmse = r2_score_and_rmse(model, processed_train_set_val, train_set_labels)

# Print the R-squared score and RMSE
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

# Predict the prices
predictions = model.predict(processed_train_set_val)

# Select the first column to visualize
x = processed_train_set_val[:, 0].toarray().flatten()  # Convert CSR matrix to 1D array

# Plot scatter of the actual data
plt.scatter(x, train_set_labels, color='black', alpha=0.6, label="Actual Data")

# Plot scatter of predictions instead of connecting them with lines
plt.scatter(x, predictions, color='blue', alpha=0.6, label="Predicted Data")

# Adding labels and legend
plt.xlabel('Feature (First column of processed_train_set_val)')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot predicted prices vs actual prices
plt.scatter(train_set_labels, predictions, color='blue', alpha=0.5)
plt.plot([min(train_set_labels), max(train_set_labels)], [min(train_set_labels), max(train_set_labels)], color='red', lw=2)  # Line y=x
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Prices')
plt.show()

# Calculate residuals
residuals = train_set_labels - predictions

# Plot residuals
plt.scatter(predictions, residuals, color='blue', alpha=0.5)
plt.axhline(y=0, color='red', lw=2)  # Zero error line
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals vs Predicted Prices')
plt.show()


print("\nInput data: \n", train_set.iloc[0:9]) # Print the first 9 rows of the input training data

# Print the predictions of the model for the first 9 samples, rounded to 1 decimal place
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))

print("Labels:      ", list(train_set_labels[0:9])) # Print the true labels for the first 9 samples

# Save the specified model to a file using the store_model function
# Uncomment the following line to save the model to a file
# store_model(model)




model = Lasso(alpha=0.5, max_iter=20000) # Create a LassoRegression model
model.fit(processed_train_set_val, train_set_labels) # Train the model on the training data

# Call the function with the model, training data, and labels, and get R2 score and RMSE
r2score, rmse = r2_score_and_rmse(model, processed_train_set_val, train_set_labels)

# Print the R-squared score and RMSE
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

# Predict the prices
predictions = model.predict(processed_train_set_val)

# Select the first column to visualize
x = processed_train_set_val[:, 0].toarray().flatten()  # Convert CSR matrix to 1D array

# Plot scatter of the actual data
plt.scatter(x, train_set_labels, color='black', alpha=0.6, label="Actual Data")

# Plot scatter of predictions instead of connecting them with lines
plt.scatter(x, predictions, color='blue', alpha=0.6, label="Predicted Data")

# Adding labels and legend
plt.xlabel('Feature (First column of processed_train_set_val)')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot predicted prices vs actual prices
plt.scatter(train_set_labels, predictions, color='blue', alpha=0.6)
plt.plot([min(train_set_labels), max(train_set_labels)], [min(train_set_labels), max(train_set_labels)], color='red', lw=2)  # Line y=x
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Prices')
plt.show()

# Calculate residuals
residuals = train_set_labels - predictions

# Plot residuals
plt.scatter(predictions, residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='red', lw=2)  # Zero error line
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals vs Predicted Prices')
plt.show()

print("\nInput data: \n", train_set.iloc[0:9]) # Print the first 9 rows of the input training data

# Print the predictions of the model for the first 9 samples, rounded to 1 decimal place
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))

print("Labels:      ", list(train_set_labels[0:9])) # Print the true labels for the first 9 samples

# Save the specified model to a file using the store_model function
# Uncomment the following line to save the model to a file
store_model(model)




model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42) # Create a Elastic_Net model
model.fit(processed_train_set_val, train_set_labels) # Train the model on the training data

# Print the learned parameters of the model
print('\n____________ Elastic_Net ____________')
print('Learned parameters: ', model.coef_, model.intercept_)

# Call the function with the model, training data, and labels, and get R2 score and RMSE
r2score, rmse = r2_score_and_rmse(model, processed_train_set_val, train_set_labels)

# Print the R-squared score and RMSE
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

# Predict the prices
predictions = model.predict(processed_train_set_val)

# Select the first column to visualize
x = processed_train_set_val[:, 0].toarray().flatten()  # Convert CSR matrix to 1D array

# Plot scatter of the actual data
plt.scatter(x, train_set_labels, color='black', alpha=0.6, label="Actual Data")

# Plot scatter of predictions instead of connecting them with lines
plt.scatter(x, predictions, color='blue', alpha=0.6, label="Predicted Data")

# Adding labels and legend
plt.xlabel('Feature (First column of processed_train_set_val)')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot predicted prices vs actual prices
plt.scatter(train_set_labels, predictions, color='blue', alpha=0.6)
plt.plot([min(train_set_labels), max(train_set_labels)], [min(train_set_labels), max(train_set_labels)], color='red', lw=2)  # Line y=x
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Prices')
plt.show()


# Calculate residuals
residuals = train_set_labels - predictions

# Plot residuals
plt.scatter(predictions, residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='red', lw=2)  # Zero error line
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals vs Predicted Prices')
plt.show()


print("\nInput data: \n", train_set.iloc[0:9]) # Print the first 9 rows of the input training data

# Print the predictions of the model for the first 9 samples, rounded to 1 decimal place
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))

print("Labels:      ", list(train_set_labels[0:9])) # Print the true labels for the first 9 samples

# Save the specified model to a file using the store_model function
# Uncomment the following line to save the model to a file
# store_model(model)




model = DecisionTreeRegressor(max_depth=10, min_samples_split=5, min_samples_leaf=4, random_state=42) # Create a DecisionTreeRegressor model
clf = model.fit(processed_train_set_val, train_set_labels) # Train the model on the training data
clf

# Call the function with the model, training data, and labels, and get R2 score and RMSE
r2score, rmse = r2_score_and_rmse(model, processed_train_set_val, train_set_labels)

# Print the R-squared score and RMSE
print('\n____________ DecisionTreeRegressor ____________')
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

print("\nInput data: \n", train_set.iloc[0:9]) # Print the first 9 rows of the input training data

# Print the predictions of the model for the first 9 samples, rounded to 1 decimal place
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))

print("Labels:      ", list(train_set_labels[0:9])) # Print the true labels for the first 9 samples

# [1]. Visualize Important Features
# If processed_train_set_val is a sparse matrix, create feature names list
feature_names = np.array([f'Feature_{i}' for i in range(processed_train_set_val.shape[1])])

# Calculate the importance of features
feature_importances = model.feature_importances_

# Sort by importance
indices = np.argsort(feature_importances)[::-1]

# Select top features (e.g., top 20)
top_n = 20  # Adjust as needed
indices = indices[:top_n]
importances = feature_importances[indices]
feature_names = feature_names[indices]

# Plot
plt.figure(figsize=(10, 8))
plt.barh(range(len(importances)), importances, align='center')
plt.yticks(range(len(importances)), feature_names)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Decision Tree')
plt.gca().invert_yaxis()  # Most important at the top
plt.show()

# Save the plot to the folder
plt.savefig('figures/feature_importance.png', bbox_inches='tight')
plt.show()

# Save the specified model to a file using the store_model function
# Uncomment the following line to save the model to a file
# store_model(model)





model = RandomForestRegressor() # Create a RandomForestRegressor model
model.fit(processed_train_set_val, train_set_labels) # Train the model on the training data

# Call the function with the model, training data, and labels, and get R2 score and RMSE
r2score, rmse = r2_score_and_rmse(model, processed_train_set_val, train_set_labels)

# Print the R-squared score and RMSE
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

print("\nInput data: \n", train_set.iloc[0:9]) # Print the first 9 rows of the input training data

# Print the predictions of the model for the first 9 samples, rounded to 1 decimal place
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))

print("Labels:      ", list(train_set_labels[0:9])) # Print the true labels for the first 9 samples

# Save the specified model to a file using the store_model function
# Uncomment the following line to save the model to a file

# store_model(model)





model = AdaBoostRegressor(n_estimators=300, learning_rate=0.05, random_state=42) # Create a LinearRegression model
model.fit(processed_train_set_val, train_set_labels) # Train the model on the training data

# Call the function with the model, training data, and labels, and get R2 score and RMSE
r2score, rmse = r2_score_and_rmse(model, processed_train_set_val, train_set_labels)

# Print the R-squared score and RMSE
print('\n____________ AdaBoostRegressor ____________')
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

# [1]. Visualize Important Features
# If processed_train_set_val is a sparse matrix, create feature names list
feature_names = np.array([f'Feature_{i}' for i in range(processed_train_set_val.shape[1])])

# Calculate the importance of features
feature_importances = model.feature_importances_

# Sort by importance
indices = np.argsort(feature_importances)[::-1]

# Select top features (e.g., top 20)
top_n = 20  # Adjust as needed
indices = indices[:top_n]
importances = feature_importances[indices]
feature_names = feature_names[indices]

# Plot
plt.figure(figsize=(10, 8))
plt.barh(range(len(importances)), importances, align='center')
plt.yticks(range(len(importances)), feature_names)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Decision Tree')
plt.gca().invert_yaxis()  # Most important at the top
plt.show()

# Save the plot to the folder
# plt.savefig('figures/feature_importance.png', bbox_inches='tight')
plt.show()

# Plot predicted prices vs actual prices
plt.scatter(train_set_labels, predictions, color='blue', alpha=0.6)
plt.plot([min(train_set_labels), max(train_set_labels)], [min(train_set_labels), max(train_set_labels)], color='red', lw=2)  # Line y=x
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Prices')
plt.show()


print("\nInput data: \n", train_set.iloc[0:9]) # Print the first 9 rows of the input training data

# Print the predictions of the model for the first 9 samples, rounded to 1 decimal place
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))

print("Labels:      ", list(train_set_labels[0:9])) # Print the true labels for the first 9 samples

# Save the specified model to a file using the store_model function
# Uncomment the following line to save the model to a file
# store_model(model)




model = GradientBoostingRegressor(random_state=42) # Create a LinearRegression model
model.fit(processed_train_set_val, train_set_labels) # Train the model on the training data

# Call the function with the model, training data, and labels, and get R2 score and RMSE
r2score, rmse = r2_score_and_rmse(model, processed_train_set_val, train_set_labels)

# Print the R-squared score and RMSE
print('\n__________ GradientBoostingRegressor __________')
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

# Predict the prices
predictions = model.predict(processed_train_set_val)

# Select the first column to visualize
x = processed_train_set_val[:, 0].toarray().flatten()  # Convert CSR matrix to 1D array

# Plot scatter of the actual data
plt.scatter(x, train_set_labels, color='black', alpha=0.6, label="Actual Data")

# Plot scatter of predictions instead of connecting them with lines
plt.scatter(x, predictions, color='blue', alpha=0.6, label="Predicted Data")

# Adding labels and legend
plt.xlabel('Feature (First column of processed_train_set_val)')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot predicted prices vs actual prices
plt.scatter(train_set_labels, predictions, color='blue', alpha=0.6)
plt.plot([min(train_set_labels), max(train_set_labels)], [min(train_set_labels), max(train_set_labels)], color='red', lw=2)  # Line y=x
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Prices')
plt.show()


# Calculate residuals
residuals = train_set_labels - predictions

# Plot residuals
plt.scatter(predictions, residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='red', lw=2)  # Zero error line
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals vs Predicted Prices')
plt.show()


print("\nInput data: \n", train_set.iloc[0:9]) # Print the first 9 rows of the input training data

# Print the predictions of the model for the first 9 samples, rounded to 1 decimal place
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))

print("Labels:      ", list(train_set_labels[0:9])) # Print the true labels for the first 9 samples

print("\nInput data: \n", train_set.iloc[0:9]) # Print the first 9 rows of the input training data

# Print the predictions of the model for the first 9 samples, rounded to 1 decimal place
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))

print("Labels:      ", list(train_set_labels[0:9])) # Print the true labels for the first 9 samples

# Save the specified model to a file using the store_model function
# Uncomment the following line to save the model to a file
# store_model(model)



from sklearn.model_selection import cross_val_score  # Import cross_val_score for cross-validation

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)  # Create a KFold object for 5-fold cross-validation
  # n_splits=5: Split data into 5 folds
  # shuffle=True: Shuffle data before splitting
  # random_state=42: Set seed for reproducibility
  
# Evaluate LinearRegression
model_name = "LinearRegression"  # Define the model name
model = LinearRegression()  # Create a LinearRegression model

# Perform cross-validation and compute negative mean squared error
nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=k_fold, scoring='neg_mean_squared_error')  # Perform cross-validation with 5 folds

rmse_scores = np.sqrt(-nmse_scores)  # Convert negative MSE to RMSE
joblib.dump(rmse_scores, 'Saved_objects/' + model_name + '_rmse.pkl')  # Save RMSE scores to a file
print("LinearRegression rmse: ", rmse_scores.round(decimals=1))  # Print RMSE scores
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')  # Print average RMSE

# Evaluate Ridge Regression
model_name = "Ridge"
model = Ridge()  # Initialize Ridge model with alpha parameter. Adjust alpha as needed.
nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=k_fold, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-nmse_scores)  # Convert negative mean squared error to root mean squared error
joblib.dump(rmse_scores, 'Saved_objects/' + model_name + '_rmse.pkl')  # Save RMSE scores to a file
print("Ridge rmse: ", rmse_scores.round(decimals=1))  # Print RMSE scores rounded to one decimal place
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')  # Print average RMSE

# Evaluate Lasso Regression
model_name = "Lasso"
model = Lasso(alpha=0.5, max_iter=20000)  # Initialize Lasso model with alpha parameter. Adjust alpha as needed.
nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=k_fold, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-nmse_scores)  # Convert negative mean squared error to root mean squared error
joblib.dump(rmse_scores, 'Saved_objects/' + model_name + '_rmse.pkl')  # Save RMSE scores to a file
print("Lasso rmse: ", rmse_scores.round(decimals=1))  # Print RMSE scores rounded to one decimal place
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')  # Print average RMSE

# Evaluate Elastic Net Regression
model_name = "ElasticNet"
model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)  # Initialize Elastic Net model with alpha and l1_ratio parameters. Adjust alpha and l1_ratio as needed.
nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=k_fold, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-nmse_scores)  # Convert negative mean squared error to root mean squared error
joblib.dump(rmse_scores, 'Saved_objects/' + model_name + '_rmse.pkl')  # Save RMSE scores to a file
print("ElasticNet rmse: ", rmse_scores.round(decimals=1))  # Print RMSE scores rounded to one decimal place
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')  # Print average RMSE

# Evaluate DecisionTreeRegressor
model_name = "DecisionTreeRegressor"  # Define the model name
model = DecisionTreeRegressor()  # Create a DecisionTreeRegressor model

# Perform cross-validation and compute negative mean squared error
nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=k_fold, scoring='neg_mean_squared_error')  # Perform cross-validation with 5 folds

rmse_scores = np.sqrt(-nmse_scores)  # Compute RMSE from negative MSE scores
joblib.dump(rmse_scores, 'Saved_objects/' + model_name + '_rmse.pkl')  # Save RMSE scores to a file
print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))  # Print RMSE scores
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')  # Print average RMSE

# Evaluate RandomForestRegressor
model_name = "RandomForestRegressor"  # Define the model name
model = RandomForestRegressor(n_estimators=5)  # Create a RandomForestRegressor model with 5 trees

# Perform cross-validation and compute negative mean squared error
nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=k_fold, scoring='neg_mean_squared_error')  # Perform cross-validation with 5 folds

rmse_scores = np.sqrt(-nmse_scores)  # Compute RMSE from negative MSE scores
joblib.dump(rmse_scores, 'Saved_objects/' + model_name + '_rmse.pkl')  # Save RMSE scores to a file
print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))  # Print RMSE scores
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')  # Print average RMSE

# Initialize AdaBoostRegressor
model_name = "AdaBoostRegressor"
model = AdaBoostRegressor(n_estimators=300, learning_rate=0.05, random_state=42)  # Adjust hyperparameters as needed

# Perform cross-validation and compute negative mean squared error
nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=k_fold, scoring='neg_mean_squared_error')  # Perform cross-validation with 5 folds

rmse_scores = np.sqrt(-nmse_scores)  # Convert negative MSE to RMSE
joblib.dump(rmse_scores, 'Saved_objects/' + model_name + '_rmse.pkl')  # Save RMSE scores to a file
print("AdaBoostRegressor rmse: ", rmse_scores.round(decimals=1))  # Print RMSE scores
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')  # Print average RMSE

# Evaluate GradientBoostingRegressor
model_name = "GradientBoostingRegressor"  # Define the model name
model = GradientBoostingRegressor(random_state=42)  # Create a GradientBoostingRegressor model

# Perform cross-validation and compute negative mean squared error
nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=k_fold, scoring='neg_mean_squared_error')  # Perform cross-validation with 5 folds

rmse_scores = np.sqrt(-nmse_scores)  # Convert negative MSE to RMSE
joblib.dump(rmse_scores, 'Saved_objects/' + model_name + '_rmse.pkl')  # Save RMSE scores to a file
print("GradientBoostingRegressor rmse: ", rmse_scores.round(decimals=1))  # Print RMSE scores
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')  # Print average RMSE




from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
# Set NumPy print options for legacy formatting
np.set_printoptions(legacy='1.25')

def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    #print('Best estimator: ', grid_search.best_estimator_) # NOTE: require refit=True in  SearchCV
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params) 
        
# Initialize variables to track the best model and score
best_score = float('-inf')  # Initialize best_score to a very low value
best_model = None           # To store the best model
best_model_index = -1       # To store the index of the best model run

# RandomForestRegressor
rf_param_dist = {
    'n_estimators': randint(120, 250),  # Number of trees
    'max_depth': randint(10, 20)  # Maximum depth of the trees
}

for i in range(3):
    print(f"Run {i + 1}:")
    random_state = 42 + i  # Change random_state each run
    rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=random_state), rf_param_dist, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1)
    rf_search.fit(processed_train_set_val, train_set_labels)
    
    # Compare and update the best model if the current one is better
    if rf_search.best_score_ > best_score:
        best_score = rf_search.best_score_  # Update best_score
        best_model = rf_search              # Update best_model
        best_model_index = i + 1               # Store the index of the best model run
        
    print("Best parameters for RandomForestRegressor:", rf_search.best_params_)
    print("Best score for RandomForestRegressor:", -rf_search.best_score_)
    print("\n" + "="*50 + "\n")
    
# Save the best model from the three runs
if best_model is not None:
    joblib.dump(best_model, 'saved_objects/RandomForestRegressor_ridgesearch.pkl')
    print_search_result(rf_search, model_name = "RandomForestRegressor")
    
# Initialize variables to track the best model and score
best_score = float('-inf')  # Initialize best_score to a very low value
best_model = None           # To store the best model
best_model_index = -1       # To store the index of the best model run

# GradientBoostingRegressor
gb_param_dist = {
    'n_estimators': randint(120, 220),  # Number of trees
    'learning_rate': uniform(0.01, 0.1),  # Learning rate
    'max_depth': randint(3, 8)  # Maximum depth of the trees
}

for i in range(3):
    print(f"Run {i + 1}:")
    random_state = 42 + i  # Change random_state each run
    gb_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=random_state), gb_param_dist, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1)
    gb_search.fit(processed_train_set_val, train_set_labels)
    
    # Compare and update the best model if the current one is better
    if gb_search.best_score_ > best_score:
        best_score = gb_search.best_score_  # Update best_score
        best_model = gb_search              # Update best_model
        best_model_index = i + 1               # Store the index of the best model run
        
    print("Best parameters for GradientBoostingRegressor:", gb_search.best_params_)
    print("Best score for GradientBoostingRegressor:", -gb_search.best_score_)
    print("\n" + "="*50 + "\n")
    
# Save the best model from the three runs
if best_model is not None:
    joblib.dump(best_model, 'saved_objects/GradientBoostingRegressor_ridgesearch.pkl')
    print_search_result(gb_search, model_name = "GradientBoostingRegressor")
    
# Initialize variables to track the best model and score
best_score = float('-inf')  # Initialize best_score to a very low value
best_model = None           # To store the best model
best_model_index = -1       # To store the index of the best model run

ridge_param_dist = {
    'alpha': uniform(1, 1.8)  
}
    
# Ridge Regression
for i in range(3):
    print(f"Run {i + 1}:")
    random_state = 42 + i  # Change random_state each run
    ridge_search = RandomizedSearchCV(Ridge(random_state=random_state), ridge_param_dist, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', random_state=random_state, n_jobs=-1)
    
    # Fit the model to the training data
    ridge_search.fit(processed_train_set_val, train_set_labels)
    
    # Compare and update the best model if the current one is better
    if ridge_search.best_score_ > best_score:
        best_score = ridge_search.best_score_  # Update best_score
        best_model = ridge_search              # Update best_model
        best_model_index = i + 1               # Store the index of the best model run
    
    # Print the best parameters and score for the current model
    print("Best parameters for Ridge:", ridge_search.best_params_)
    print("Best score for Ridge:", -ridge_search.best_score_)  # Negate the score for RMSE
    print("\n" + "="*50 + "\n")

# Save the best model from the three runs
if best_model is not None:
    joblib.dump(best_model, 'Saved_objects/Ridge_ridgesearch.pkl')
    print_search_result(ridge_search, model_name = "Ridge")
    



# Dictionary to store model names and their respective file paths
models_info = {
    'Ridge': 'Saved_objects/Ridge_ridgesearch.pkl',
    'RandomForest': 'Saved_objects/RandomForestRegressor_ridgesearch.pkl',
    'GradientBoosting': 'Saved_objects/GradientBoostingRegressor_ridgesearch.pkl'
}

from sklearn.metrics import mean_absolute_error

def evaluate_model_performance(model, model_name, processed_test_set, test_set_labels):
    print(f'\nTesting {model_name}...')

    # Calculate R², MAE, MSE, RMSE
    r2score = model.score(processed_test_set, test_set_labels)
    predictions = model.predict(processed_test_set)
    mae = mean_absolute_error(test_set_labels, predictions)
    mse = mean_squared_error(test_set_labels, predictions)
    rmse = np.sqrt(mse)
    
    # Print out performance metrics
    print(f"Performance on test data for {model_name}:")
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Square Error (RMSE):", rmse)
    print('R² score (on test data, best=1):', r2score)
    
    # Print out first 9 rows of test data, predictions, and actual labels
    print("\nTest data (first 9 rows):\n", test_set.iloc[0:9])
    print(f"\n{model_name} Predictions (first 9 rows):", predictions[0:9].round(decimals=1))
    print("True Labels (first 9 rows):", list(test_set_labels[0:9]), '\n')
    
# Load the full preprocessing pipeline
full_pipeline = joblib.load('Models/full_pipeline.pkl')

# Preprocess the test set using the saved pipeline
processed_test_set = full_pipeline.transform(test_set)  

print('\n____________ ANALYZE AND TEST YOUR SOLUTION __________')

# Loop through each model
for model_name, model_path in models_info.items():
    # Load the saved model
    search = joblib.load(model_path)
    best_model = search.best_estimator_
    print('SOLUTION: ' , best_model)

    # Gain insights and print features (e.g., coefficients or feature importances)
    if model_name == 'Ridge':
        coefficients = best_model.coef_
        # Process the feature names (combine binarized and OneHot features)
        onehot_cols = []
        for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_:
            onehot_cols += val_list.tolist()
        feature_names = train_set.columns.tolist() + ["power_to_displacement", "mileage_per_day", "average_fuel_economy"] + onehot_cols
        for name in categorial_feat_names:
            if name in feature_names:
                feature_names.remove(name)
        feature_importances = sorted(zip(feature_names, coefficients.round(decimals=4)), key=lambda row: abs(row[1]), reverse=True)
        print("Top 10 features by coefficients in Ridge Regression:")
        for feature, coef in feature_importances[:10]:
            print(f"{feature}: {coef}")
    elif model_name in ['RandomForest', 'GradientBoosting']:
        feature_importances = best_model.feature_importances_
        # Process the feature names for RandomForest and GradientBoosting
        onehot_cols = []
        for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_:
            onehot_cols += val_list.tolist()
        feature_names = train_set.columns.tolist() + ["power_to_displacement", "mileage_per_day", "average_fuel_economy"] + onehot_cols
        for name in categorial_feat_names:
            if name in feature_names:
                feature_names.remove(name)
        feature_importances = sorted(zip(feature_names, feature_importances.round(decimals=4)), key=lambda row: row[1], reverse=True)
        print(f"Top 10 features by importance in '{model_name}':")
        for feature, importance in feature_importances[:10]:
            print(f"{feature}: {importance}")

    # Evaluate and print performance metrics
    evaluate_model_performance(best_model, model_name, processed_test_set, test_set_labels)
    
