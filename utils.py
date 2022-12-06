from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

# utils

def metrics(test_y, y_preds):
    mse = mean_squared_error(test_y, y_preds, squared=False)
    rmsle = mean_squared_log_error(test_y, y_preds, squared=False)
    r2 = r2_score(test_y, y_preds)
    return mse, rmsle, r2