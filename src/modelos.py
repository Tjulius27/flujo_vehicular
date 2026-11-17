"""Modelos de prediccion del volumen vehicular mensual (indice INEI de flujo en peajes)."""
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

SEMILLA = 42
PANDEMIA = ("2020-03-01", "2021-04-01")   # meses atipicos que se reemplazan al entrenar
MODELOS = ["Media móvil", "Método estacional ingenuo", "Holt-Winters", "SARIMA",
           "Random Forest", "Red neuronal multicapa"]
COLUMNAS = ["lag1", "lag2", "lag3", "lag12", "mes_sen", "mes_cos"]
UN_MES = pd.DateOffset(months=1)


# ---------------------------------------------------------------- datos
def cargar_serie(ruta, columna="total"):
    df = pd.read_csv(ruta, parse_dates=["fecha"]).set_index("fecha")
    return df[columna].astype(float).asfreq("MS")


def limpiar_pandemia(s):
    """Reemplaza cada mes de pandemia por el promedio del mismo mes en los años (±3) no afectados."""
    s = s.copy()
    marca = (s.index >= PANDEMIA[0]) & (s.index <= PANDEMIA[1])
    for fecha in s.index[marca]:
        mismos = s[(s.index.month == fecha.month) & (~marca) & (abs(s.index.year - fecha.year) <= 3)]
        s.loc[fecha] = mismos.mean()
    return s


def variacion_interanual(s):
    """Ecuación 2: variación porcentual respecto al mismo mes del año anterior."""
    return ((s - s.shift(12)) / s.shift(12) * 100).dropna().round(1)


def nivel_de_demanda(entrenamiento, valores):
    """Ecuación 5: bajo / medio / alto según los terciles de los meses de entrenamiento."""
    u1, u2 = entrenamiento.quantile([1 / 3, 2 / 3])
    return pd.cut(valores, [-np.inf, u1, u2, np.inf], labels=["bajo", "medio", "alto"]), (u1, u2)


# ---------------------------------------------------------------- variables para ML
def fila(historia, fecha):
    """Variables de entrada para pronosticar el mes `fecha` (la historia termina el mes anterior)."""
    m = fecha.month
    return [historia.iloc[-1], historia.iloc[-2], historia.iloc[-3], historia.iloc[-12],
            np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]


def tabla_ml(s):
    filas, y = [], []
    for i in range(12, len(s)):
        filas.append(fila(s.iloc[:i], s.index[i]))
        y.append(s.iloc[i])
    return pd.DataFrame(filas, columns=COLUMNAS, index=s.index[12:]), pd.Series(y, index=s.index[12:])


def crear_ml(nombre):
    if nombre == "Random Forest":
        return RandomForestRegressor(n_estimators=300, random_state=SEMILLA)
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(),
                                MLPRegressor(hidden_layer_sizes=(16,), max_iter=5000, random_state=SEMILLA)),
        transformer=StandardScaler())


def entrenar_ml(nombre, historia):
    X, y = tabla_ml(historia)
    modelo = crear_ml(nombre)
    modelo.fit(X, y)
    return modelo


def pronosticar_ml(modelo, historia, h):
    valores = historia.copy()
    for _ in range(h):
        fecha = valores.index[-1] + UN_MES
        x = pd.DataFrame([fila(valores, fecha)], columns=COLUMNAS)
        valores.loc[fecha] = float(modelo.predict(x)[0])
    return valores.iloc[-h:]


# ---------------------------------------------------------------- pronostico general
def pronosticar(nombre, historia, h=1):
    """Pronostica los próximos h meses con el modelo indicado, usando solo `historia`."""
    if nombre in ("Media móvil", "Método estacional ingenuo"):
        valores = historia.copy()
        for _ in range(h):
            fecha = valores.index[-1] + UN_MES
            valores.loc[fecha] = valores.iloc[-3:].mean() if nombre == "Media móvil" else valores.iloc[-12]
        return valores.iloc[-h:]
    if nombre == "Holt-Winters":
        m = ExponentialSmoothing(historia, trend="add", damped_trend=True, seasonal="add",
                                 seasonal_periods=12, initialization_method="estimated").fit()
        return m.forecast(h)
    if nombre == "SARIMA":
        m = SARIMAX(historia, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return m.forecast(h)
    return pronosticar_ml(entrenar_ml(nombre, historia), historia, h)


# ---------------------------------------------------------------- evaluación
def evaluar(s, n_prueba=12):
    """Validación cronológica: se entrena con los meses antiguos y se prueba con los últimos n_prueba.
    Cada mes de prueba se predice usando solo los meses anteriores (un paso adelante)."""
    entrenamiento, prueba = s.iloc[:-n_prueba], s.iloc[-n_prueba:]
    pred = {"Real": prueba}
    for nombre in MODELOS:
        if nombre in ("Random Forest", "Red neuronal multicapa"):
            modelo = entrenar_ml(nombre, entrenamiento)
            vals = [modelo.predict(pd.DataFrame([fila(s.loc[: f - UN_MES], f)], columns=COLUMNAS))[0]
                    for f in prueba.index]
        else:
            vals = [pronosticar(nombre, s.loc[: f - UN_MES], 1).iloc[0] for f in prueba.index]
        pred[nombre] = pd.Series(vals, index=prueba.index)
    pred = pd.DataFrame(pred)

    niveles_real, _ = nivel_de_demanda(entrenamiento, pred["Real"])
    filas = []
    for nombre in MODELOS:
        e = pred["Real"] - pred[nombre]
        niv, _ = nivel_de_demanda(entrenamiento, pred[nombre])
        filas.append({"Modelo": nombre,
                      "MAE": e.abs().mean(),
                      "RMSE": np.sqrt((e ** 2).mean()),
                      "MAPE (%)": (e.abs() / pred["Real"].abs()).mean() * 100,
                      "Aciertos de nivel (%)": (niv.values == niveles_real.values).mean() * 100})
    return pred.round(2), pd.DataFrame(filas).round(2)