import pandas as pd 
import numpy as np
from sklearn import preprocessing
import src.time_series_functions as tsf


def fit_sklearn_model(ts, model, test_size, val_size):
    """
     Parâmetros:
    > ts (pandas.DataFrame): valores de série temporal criados por
                            src.time_series_functions.create_window
    > modelo (Sklearn Model): modelo base para prever ts
    > test_size (int): tamanho do conjunto de teste
    > val_size (int): tamanho do conjunto de validação (se você não usar o conjunto de validação,
                      val_size pode ser definido como 0)

     Devoluções:
    > Modelo Sklearn: modelo treinado 
    """


    train_size = len(ts) - test_size - val_size
    y_train = ts['actual'][0:train_size]
    x_train = ts.drop(columns=['actual'], axis=1)[0:train_size]

    return model.fit(x_train, y_train)


def predict_sklearn_model(ts, model):
    """
    Parâmetros:
    > ts (pandas.DataFrame): valores de série temporal criados por
                            src.time_series_functions.create_window
    > modelo (Sklearn Model): modelo base para prever ts
    
     Devoluções:
     > matriz numpy: valores previstos
    """

    x = ts.drop(columns=['actual'], axis=1)
    return model.predict(x)

def additive_hybrid_model(predicted, real, time_window, base_model,
                          test_size, val_size):
    """
   Parâmetros:
    > real (matriz numpy): valores reais da série temporal
    > previsto (matriz numpy): previsão linear da série temporal
    > time_window (int): janela de tempo do modelo de previsão de erro
    > base_model (modelo Sklearn): modelo base para prever o erro
    > test_size (int): tamanho do conjunto de teste
    > val_size (int): tamanho do conjunto de validação (se você não usar o conjunto de validação,
                      val_size pode ser definido como 0)

     Devoluções:
     > dict: seguindo o formato src.time_series_functions.make_metrics_avaliation 
    """

    train_size = len(predicted) - test_size

    errors = np.subtract(real, predicted)

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(errors[0:train_size].reshape(-1, 1))
    normalized_error = min_max_scaler.transform(errors.reshape(-1, 1))

    # fit_predict

    error_values = pd.DataFrame({'actual': normalized_error.flatten()})

    error_windowed = tsf.create_windowing( df=error_values, lag_size=time_window)

    pi = fit_sklearn_model(ts=error_windowed, model=base_model,
                               test_size=test_size, val_size=val_size)

    pi_pred = predict_sklearn_model(ts=error_windowed, model=pi)
    # _____________________________

    pi_pred = min_max_scaler.inverse_transform(pi_pred.reshape(-1, 1)).flatten()

    prevs = predicted[time_window:] + pi_pred

    ts_actual = real[time_window:]

    return tsf.make_metrics_avaliation(ts_actual, prevs, test_size,
                                       val_size, base_model.get_params(deep=True))

def format_nolic_input(real, nonlinear_forecast, linear_forecast, test_size, time_window):
    """
    Parâmetros:
    > real (matriz numpy): valores reais de séries temporais
    > linear_forecast (matriz numpy): previsão linear da série temporal
    > nonlinear_forecast (numpy array): previsão de sistema híbrido aditivo
    > time_window (int): janela de tempo do modelo de previsão de erro
    > test_size (int): tamanho do conjunto de teste

     Devoluções:
    > pandas DataFrame: formato de entrada x saída
    > pré-processamento sklearn: modelo de normalização de dado
    """
    train_size_represents = len(real) - test_size
    
    error_values = nonlinear_forecast - linear_forecast

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(real[0:train_size_represents].reshape(-1, 1))

    min_max_scaler_linear = preprocessing.MinMaxScaler()
    min_max_scaler_linear.fit(linear_forecast[0:train_size_represents].reshape(-1, 1))

    min_max_scaler_error = preprocessing.MinMaxScaler()
    min_max_scaler_error.fit(error_values[0:train_size_represents].reshape(-1, 1))

    real_normalized = min_max_scaler.transform(real.reshape(-1, 1)).flatten()
    linear_normalized = min_max_scaler_linear.transform(linear_forecast.reshape(-1, 1)).flatten()
    error_normalized = min_max_scaler_error.transform(error_values.reshape(-1, 1)).flatten()
      
    tsf_part = tsf.create_windowing(lag_size=(time_window - 1),
                                    df=pd.DataFrame({'actual': linear_normalized}))

    ef_part = tsf.create_windowing(lag_size=(time_window - 1),
                                   df=pd.DataFrame({'actual': error_normalized}))

    real_part = tsf.create_windowing(lag_size=(time_window - 1),
                                     df=pd.DataFrame({'actual': real_normalized}))

    tsf_part.columns = ['ts_prev' + str(i) for i in reversed(range(0, time_window))]
    ef_part.columns = ['error_prev' + str(i) for i in reversed(range(0, time_window))]

    ts_formated = pd.concat([ef_part, tsf_part,
                             real_part['actual']], axis=1)
    return ts_formated, min_max_scaler

def nolic_model(linear_forecast, real, nonlinear_forecast,time_window, 
                base_model, test_size,val_size):

    """
    Parâmetros:
    > real (matriz numpy): valores reais da série temporal
    > linear_forecast (matriz numpy): previsão linear da série temporal
    > nonlinear_forecast (numpy array): previsão de sistema híbrido aditivo
    > time_window (int): janela de tempo do modelo de previsão de erro
    > base_model (modelo Sklearn): modelo base para combinar a predição linear e não linear
    > test_size (int): tamanho do conjunto de teste
    > val_size (int): tamanho do conjunto de validação (se você não usar o conjunto de validação,
                      val_size pode ser definido como 0)

     Devoluções:
    > dict: seguindo o formato src.time_series_functions.make_metrics_avaliation 
    """

    ts_formated, min_max_scaler= format_nolic_input(real, nonlinear_forecast, linear_forecast, test_size, time_window)

    p = fit_sklearn_model(ts=ts_formated, model=base_model,
                              test_size=test_size, val_size=val_size)

    pred = predict_sklearn_model(ts=ts_formated,
                                     model=p)

    pred = min_max_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

    real_actual = real[(time_window - 1):]
    
    result_metrics = tsf.make_metrics_avaliation(real_actual, pred, test_size,
                                       val_size, base_model.get_params(deep=True))
    return result_metrics
