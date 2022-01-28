#  Data Preprocessing
import opendatasets as od
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# Download the dataset and load the data 


print('Download the dataset and load the data') 
raw_df = pd.read_csv('D:/Machine Learing/Spyder/weatherAUS.csv')
print(raw_df)
print(raw_df.info(),raw_df.describe())

print('Removing the null value in  RainToday  RainTomorrow')
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)
print(raw_df)
print(raw_df.info(),raw_df.describe())
print("Exploratory Data Analysis and Visualization") 
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
'''
fig  =px.histogram(raw_df, x='Location', title='Location vs. Rainy Days', color='RainToday')
fig.show()
fig  =px.histogram(raw_df, x='Location', title='Location vs. Non Rainy Days', color='RainTomorrow')
fig.show()
fig  =px.histogram(raw_df, 
             x='Temp3pm', 
             title='Temperature at 3 pm vs. Rain Tomorrow', 
             color='RainTomorrow')
fig.show()
fig  =px.histogram(raw_df, 
             x='RainTomorrow', 
             color='RainToday', 
             title='Rain Tomorrow vs. Rain Today')
fig.show()

fig  =px.scatter(raw_df.sample(2000), 
           title='Min Temp. vs Max Temp.',
           x='MinTemp', 
           y='MaxTemp', 
           color='RainToday')
fig.show()
fig  =px.scatter(raw_df.sample(2000), 
           title='Temp (3 pm) vs. Humidity (3 pm)',
           x='Temp3pm',
           y='Humidity3pm',
           color='RainTomorrow')
fig.show()
'''
# sample 
# Make sure to set use_sample to False and re-run the notebook end-to-end once you're ready to use the entire dataset.
use_sample = False
sample_fraction = 0.1
if use_sample:
    raw_df = raw_df.sample(frac=sample_fraction).copy()


print("Create training, validation and test sets")
from sklearn.model_selection import train_test_split
'''
train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)
print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)'''
year = pd.to_datetime(raw_df.Date).dt.year


# However, while working with dates, it's often a better idea to separate the training, validation and test sets with time, so that the model is trained on data from the past and evaluated on data from the future.


train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]
print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)

# Identifying Input and Target Columns

print("Identifying Input and Target Columns")
input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'
print(input_cols)
print(target_col)

print("make an copys")
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()
import numpy as np
print("seprate numaric and object data")
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
print(numeric_cols)
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()
print(categorical_cols)
print(train_inputs[numeric_cols].describe())
print(train_inputs[categorical_cols].nunique())




#Imputing Missing Numeric Data



print("Imputing Missing Numeric Data  There are several techniques for imputation, but we'll use the most basic one: replacing missing values with the average value in the column using the SimpleImputer class from sklearn.impute.")

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'mean')
print(raw_df[numeric_cols].isna().sum())
print(test_inputs[numeric_cols].isna().sum())
print(train_inputs[numeric_cols].isna().sum())
print(val_inputs[numeric_cols].isna().sum())
imputer.fit(raw_df[numeric_cols])
print(list(imputer.statistics_))
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])
print(train_inputs[numeric_cols].isna().sum())

# Scaling Numeric Features
print("change data frame to The numeric columns in our dataset have varying ranges")
print(raw_df[numeric_cols].describe())
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(raw_df[numeric_cols])
print('Minimum:')
print(list(scaler.data_min_))
print('Maximum:')
print(list(scaler.data_max_))
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])
print(train_inputs[numeric_cols].describe())
print("We can perform one hot encoding using the OneHotEncoder class from sklearn.preprocessing.")
print(raw_df[categorical_cols].nunique())
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(raw_df[categorical_cols])
print(encoder.categories_)

encoded_cols = list(encoder.get_feature_names(categorical_cols))
print(encoded_cols)
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols].fillna('Unknown'))
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols].fillna('Unknown'))
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols].fillna('Unknown'))
#pd.set_option('display.max_columns', None)
print(train_inputs)
print('train_inputs:', train_inputs.shape)
print('train_targets:', train_targets.shape)
print('val_inputs:', val_inputs.shape)
print('val_targets:', val_targets.shape)
print('test_inputs:', test_inputs.shape)
print('test_targets:', test_targets.shape)
train_inputs.to_parquet('train_inputs.parquet')
val_inputs.to_parquet('val_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')
pd.DataFrame(train_targets).to_parquet('train_targets.parquet')
pd.DataFrame(val_targets).to_parquet('val_targets.parquet')
pd.DataFrame(test_targets).to_parquet('test_targets.parquet')
train_inputs = pd.read_parquet('train_inputs.parquet')
val_inputs = pd.read_parquet('val_inputs.parquet')
test_inputs = pd.read_parquet('test_inputs.parquet')

train_targets = pd.read_parquet('train_targets.parquet')[target_col]
val_targets = pd.read_parquet('val_targets.parquet')[target_col]
test_targets = pd.read_parquet('test_targets.parquet')[target_col]
print('train_inputs:', train_inputs.shape)
print('train_targets:', train_targets.shape)
print('val_inputs:', val_inputs.shape)
print('val_targets:', val_targets.shape)
print('test_inputs:', test_inputs.shape)
print('test_targets:', test_targets.shape)


print( train_inputs)
print(train_targets)
print(val_inputs)
print(val_targets)
print(test_inputs)
print(test_targets)

# Training a Logistic Regression Model
print("Training a Logistic Regression Model")
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)
print(numeric_cols + encoded_cols)
print(model.coef_.tolist())
print(model.intercept_)
# Making Predictions and Evaluating the Model
print("Making Predictions and Evaluating the Model")
X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]
train_preds = model.predict(X_train)
train_probs = model.predict_proba(X_train)
# to check Weight 
print("Weights")
n = len(model.coef_.tolist())
weights_df = pd.DataFrame({
    'feature': (numeric_cols + encoded_cols ) ,
    'weight' : (model.coef_.tolist()[0])
})
print(weights_df)
fig  =px.histogram(weights_df, 
             x='weight', 
             y='feature', 
             title='Weight vs Features')
fig.show()

print(train_preds)
print(train_targets)
print("Confidence on pridictioc")
print(train_probs)
from sklearn.metrics import accuracy_score
print(accuracy_score(train_targets, train_preds))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(train_targets, train_preds, normalize='true'))
def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)
    
    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name));
    plt.show()
    
    return preds
train_preds = predict_and_plot(X_train, train_targets, 'Training')  
val_preds = predict_and_plot(X_val, val_targets, 'Validatiaon')  
test_preds = predict_and_plot(X_test, test_targets, 'Test')
def random_guess(inputs):
    return np.random.choice(["No", "Yes"], len(inputs))
def all_no(inputs):
    return np.full(len(inputs), "No")

accuracy_score(test_targets, random_guess(X_test))    
accuracy_score(test_targets, all_no(X_test))

new_input = {'Date': '2021-06-19',
             'Location': 'Katherine',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}
new_input_df = pd.DataFrame([new_input])
print(new_input_df)

'''We must now apply the same transformations applied while training the model:

Imputation of missing values using the imputer created earlier
Scaling numerical features using the scaler created earlier
Encoding categorical features using the encoder created earlier'''

new_input_df[numeric_cols] = imputer.transform(new_input_df[numeric_cols])
new_input_df[numeric_cols] = scaler.transform(new_input_df[numeric_cols])
new_input_df[encoded_cols] = encoder.transform(new_input_df[categorical_cols])
X_new_input = new_input_df[numeric_cols + encoded_cols]
print(X_new_input)
prediction = model.predict(X_new_input)[0]
print(prediction)
prob = model.predict_proba(X_new_input)[0]
print(prob)

def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    
    return pred, prob
new_input = {'Date': '2021-06-19',
             'Location': 'Launceston',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}
print(predict_input(new_input)    )        