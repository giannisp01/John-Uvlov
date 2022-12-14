# Data manipulation
import pandas as pd

# Manipulation with dates
from datetime import date
from dateutil.relativedelta import relativedelta

# Machine learning
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np


def XGBoost(x_train, y_train, x_test, y_test):
    DM_train = xgb.DMatrix(data=x_train, label=y_train)
    grid_param = {"learning_rate": [0.01, 0.1],
                  "n_estimators": [100, 150, 200],
                  "alpha": [0.1, 0.5, 1],
                  "max_depth": [2, 3, 4]}
    model = xgb.XGBRegressor()
    grid_mse = GridSearchCV(estimator=model, param_grid=grid_param,
                            scoring="neg_mean_squared_error",
                            cv=4, verbose=1)
    grid_mse.fit(x_train, y_train)
    #print("Best parameters found: ", grid_mse.best_params_)
    #print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
    alpha, learning_rate, max_depth, n_estimators = grid_mse.best_params_['alpha'], grid_mse.best_params_['learning_rate'], grid_mse.best_params_['max_depth'], grid_mse.best_params_['n_estimators']
    RMSE = np.sqrt(np.abs(grid_mse.best_score_))
    xgb_model = xgb.XGBClassifier(objective='reg:squarederror',
                                  colsample_bytree=1,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  alpha=alpha,
                                  n_estimators=n_estimators)
    xgb_model.fit(x_train, y_train)
    xgb_prediction = xgb_model.predict(x_test)
    xgb_matrix = metrics.confusion_matrix(xgb_prediction, y_test)
    return xgb_model, xgb_matrix

def knn(x_train, y_train, x_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=3, algorithm='auto',
                               weights='distance')
    knn.fit(x_train, y_train)
    knn_prediction = knn.predict(x_test)
    knn_matrix = metrics.confusion_matrix(knn_prediction, y_test)
    return knn, knn_matrix

def RandomForest(x_train, y_train, x_test, y_test):
    random_forest = RandomForestClassifier(n_estimators=50,
                                           max_depth=10, random_state=1)
    random_forest.fit(x_train, y_train)
    rf_prediction = random_forest.predict(x_test)
    rf_matrix = metrics.confusion_matrix(rf_prediction, y_test)
    return random_forest, rf_matrix

def holiday(data, holidays):
    data_holidays = []
    for i in range(len(data['Date'])):
        day = data['Date'].dt.day[i]
        month = data['Date'].dt.month[i]
        holiday = False
        for day_holidays,month_holidays in np.array(holidays):
            if day_holidays==day and month==month_holidays:
                data_holidays.append(True)
                holiday = True
        if not holiday:
            data_holidays.append(False)
    return pd.Series(data_holidays)


data = pd.DataFrame({'Date': ['2020-12-22', '2020-11-24', '2020-10-27',
                              '2020-09-29', '2020-08-25', '2020-07-28',
                              '2020-06-30', '2020-05-26', '2020-04-28',
                              '2020-03-31', '2020-02-25', '2020-01-28',
                              '2019-12-31', '2019-11-26', '2019-10-29',
                              '2019-09-24', '2019-08-27', '2019-07-30',
                              '2019-06-25', '2019-05-28', '2019-04-30',
                              '2019-03-26', '2019-02-26', '2019-01-29',
                              '2018-12-27', '2018-11-27', '2018-10-30',
                              '2018-09-25', '2018-08-28', '2018-07-31',
                              '2018-06-26', '2018-05-29', '2018-04-24',
                              '2018-03-27', '2018-02-27', '2018-01-30',
                              '2017-12-27', '2017-11-28', '2017-10-31',
                              '2017-09-26', '2017-08-29', '2017-07-25',
                              '2017-06-27', '2017-05-30', '2017-04-25',
                              '2017-03-28', '2017-02-28', '2017-01-31',
                              '2016-12-27', '2016-11-29', '2016-10-25',
                              '2016-09-27', '2016-08-30', '2016-07-26',
                              '2016-06-28', '2016-05-31', '2016-04-26',
                              '2016-03-29', '2016-02-23', '2016-01-26',
                              '2015-12-29', '2015-11-24', '2015-10-27',
                              '2015-09-29', '2015-08-25', '2015-07-28',
                              '2015-06-30', '2015-05-26', '2015-04-28',
                              '2015-03-31', '2015-02-24', '2015-01-27',
                              '2014-12-30', '2014-11-25', '2014-10-28',
                              '2014-09-30', '2014-08-26', '2014-07-29',
                              '2014-06-24', '2014-05-27', '2014-04-29',
                              '2014-03-25', '2014-02-25', '2014-01-28',
                              '2013-12-31', '2013-11-26', '2013-10-29',
                              '2013-09-24', '2013-08-27', '2013-07-30',
                              '2013-06-25', '2013-05-28', '2013-04-30',
                              '2013-03-26', '2013-02-26', '2013-01-29']})

# Convert 'date' object to datetime type
data['Date'] = pd.to_datetime(data['Date'])
# If a release is happened at that day then release=1
data['Release'] = 1

# Adjust the data to fill all dates not only dates that en event has been released
r = pd.date_range(start=data['Date'].min(), end=data['Date'].max())
data = data.set_index('Date').reindex(r).fillna(0.0).rename_axis('Date').reset_index()

holidays = pd.read_csv('Holidays.csv')
# Create the following features for each date:
#       Month           --> Month
#       Day             --> Day (date)
#       Workday_N       --> Number of working day of the month
#       Week_day        --> Week day (0->Monday, 1->Tuesday,...)
#       Weekday_order   --> Which day of the month (e.g the 4th Tuesday of the month)
#       Holiday         --> Whether this date is a holiday (e.g 25-December --> True)
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Workday_N'] = np.busday_count(
                    data['Date'].values.astype('datetime64[M]'),
                    data['Date'].values.astype('datetime64[D]'))
data['Total_days'] = data['Date'].dt.daysinmonth - data['Date'].dt.day
data['Week_day'] = data['Date'].dt.weekday
data['Weekday_order'] = (data['Date'].dt.day + 6) // 7
data['Holiday'] = holiday(data, holidays)
data = data.set_index('Date')

#print(data)
#print()
#print(data.info())

# Training Machine Learning Model
x_train, x_test, y_train, y_test = train_test_split(data.drop(['Release'], axis=1), data['Release'],
                 test_size=0.3, random_state=1, shuffle=False)

xgb_model, xgb_confusion_matrix = XGBoost(x_train, y_train, x_test, y_test)
knn, knn_confusion_matrix = knn(x_train, y_train, x_test, y_test)
random_forest, rf_confusion_matrix = RandomForest(x_train, y_train, x_test, y_test)

print(f"""
Confusion matrix for XGBoost model:
TN:{xgb_confusion_matrix[0][0]}    FN:{xgb_confusion_matrix[0][1]}
FP:{xgb_confusion_matrix[1][0]}    TP:{xgb_confusion_matrix[1][1]}""")

print(f"""
Confusion matrix for KNN model:
TN:{knn_confusion_matrix[0][0]}    FN:{knn_confusion_matrix[0][1]}
FP:{knn_confusion_matrix[1][0]}    TP:{knn_confusion_matrix[1][1]}""")

print(f"""
Confusion matrix for Random Forest model:
TN:{rf_confusion_matrix[0][0]}    FN:{rf_confusion_matrix[0][1]}
FP:{rf_confusion_matrix[1][0]}    TP:{rf_confusion_matrix[1][1]}""")


print()
print()
print("Explanation:")
print("TN:\tPredicted-NO, Actual-NO")
print("FN:\tPredicted-NO, Actual-YES")
print("FP:\tPredicted-YES, Actual-NO")
print("TP:\tPredicted-YES, Actual-YES")


print("\t\t   Predicted")
print("\t\t\tNo\tYes")
print("\t\tYes\tTN\tFP")
print("Actual")
print("\t\tNo\tFN\tTP")
print()
print()


##################################################################
#                    Forecast for one year ahead                 #
##################################################################

# Create the table and fill it like training
x_predict = pd.DataFrame(pd.date_range(date(2017,1,1), (date(2017,1,1) +
            relativedelta(years=7)),freq='d'), columns=['Date'])
x_predict['Month'] = x_predict['Date'].dt.month
x_predict['Day'] = x_predict['Date'].dt.day
x_predict['Workday_N'] = np.busday_count(
                x_predict['Date'].values.astype('datetime64[M]'),
                x_predict['Date'].values.astype('datetime64[D]'))
x_predict['Total_days'] = x_predict['Date'].dt.daysinmonth - x_predict['Date'].dt.day
x_predict['Week_day'] = x_predict['Date'].dt.weekday
x_predict['Weekday_order'] = (x_predict['Date'].dt.day + 6) // 7
x_predict['Holiday'] = holiday(x_predict, holidays)
x_predict = x_predict.set_index('Date')

predictionrandom_forest = random_forest.predict(x_predict)
predictionxgboost = xgb_model.predict(x_predict)

print(x_predict[predictionxgboost==1])
print(x_predict[predictionrandom_forest==1])

