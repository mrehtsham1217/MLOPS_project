# from dotenv import load_dotenv
# import pymysql
# import os
#
# # Load environment variables from .env file
# load_dotenv()
#
# # Retrieve environment variables
# host = os.getenv("host")
# user = os.getenv("user")
# password = os.getenv("password")
# db = os.getenv("db")
#
# # Connect to MySQL database
# connection = pymysql.connect(
#     host=host,
#     user=user,
#     password=password,
#     db=db
# )
#
# # Create cursor
# cursor = connection.cursor()
#
# # Execute SQL query
# query = 'SELECT * FROM raw'
# cursor.execute(query)
#
# # Fetch and print rows
# rows = cursor.fetchall()
# for row in rows:
#     print(row)
#
# # Close cursor and connection
# cursor.close()
# connection.close()
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

parmas = {
    "Decision Tree": {
        'criterion': ["squared_error", "friedman_mse", "absolute_error"],
        'splitter': ["best", "random"],
        'max_features': ['sqrt', 'log2']
    },
    "Random Forest": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'max_features': ['sqrt', 'log2', None],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "Gradient Boosting": {
        'learning_rate': [.1, .01, .05, .001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "Linear Regression": {},
    "XGBRegressor": {
        'learning_rate': [.1, .01, .05, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "CatBoosting Regressor": {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    },
    "AdaBoost Regressor": {
        'learning_rate': [.1, .01, 0.5, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    }
}

models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
  }


for i in range(len(list(models))):
    model = list(models.values())[i]
    para = parmas[list(models.keys())[i]]
    print(model)
    print("*"*50)
    print(para)