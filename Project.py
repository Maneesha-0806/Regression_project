#Importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Data Loading
df = pd.read_csv("data/Indian Unicorn startups 2023 updated.csv")
df.head()

#Drop Irrelevant columns
df = df.drop(columns = ['No.', 'Company'])
#Rename columns
df = df.rename(columns = {'Entry Valuation^^ ($B)' : 'Entry_Valuation', 'Valuation ($B)' : 'Valuation'})

#Extract exact year
df['Entry_year'] = pd.to_datetime(df['Entry'], format = '%b/%Y').dt.year
df = df.drop(columns = ['Entry'])

#Investor Count feature
df['Investor_Count'] = df['Select Investors'].apply(lambda x: len(str(x).split(',')))
df = df.drop(columns = ['Select Investors'])

#Features and target
X = df.drop(columns = ['Valuation'])
y = df['Valuation']

numerical_features = ['Entry_Valuation', 'Entry_year', 'Investor_Count']
categorical_features = ['Sector', 'Location']
numeric_transformer = Pipeline(steps = [('scaler', StandardScaler())])
categorical_transformer = OneHotEncoder(drop = 'first', handle_unknown = 'ignore')
preprocessor = ColumnTransformer(transformers = [('num', numeric_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)])
#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#function for visualizing actual vs predicted
def plot_actual_vs_pred(y_test, y_pred, model_name):
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()])
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.show()

#Evaluation function for all models
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    plot_actual_vs_pred(y_test, y_test_pred, name)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)

    return{'Model' : name, 'Train RMSE' : train_rmse, 'Test RMSE' : test_rmse, 'R2 Score' : r2}

models = []
#Linear regression
models.append((
    "Linear Regression", 
    Pipeline(steps = [
        ('preprocessor', preprocessor), 
        ('model', LinearRegression())
    ])
))
#Ridge Regression
models.append((
    "Ridge Regression",
    Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha = 1.0))
    ])
))
#Lasso Regression
models.append((
    "Lasso Regression",
    Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('model', Lasso(alpha = 0.01))
    ])
))

#run evaluation for all models
results = []
for name, model in models:
    results.append(
        evaluate_model(name, model, X_train, X_test, y_train, y_test)
    )

results_df = pd.DataFrame(results)
results_df

#Visual Comparison
results_df.set_index('Model')[['Train RMSE', 'Test RMSE']].plot(kind = 'bar')
plt.ylabel("RMSE")
plt.title("Train vs Test RMSE Comparison")
plt.show()

best_model = results_df.sort_values(by = 'Test RMSE').iloc[0]
print("Best Model:", best_model)
def get_user_input():
    data = {}
    for col in X.columns:
        value = input(f"Enter value for {col}: ")
        data[col] = [value]
    return pd.DataFrame(data)    
user_input = get_user_input()
for name, model in models:
    prediction = model.predict(user_input)[0]
    print(f"{name} Prediction: {prediction:.2f}")
