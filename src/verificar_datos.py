"""Comprueba la transcripción: total ≈ 0,556 × pesados + 0,444 × ligeros (ponderaciones del INEI)."""
import pandas as pd

df = pd.read_csv("data/indice_flujo_vehicular.csv", parse_dates=["fecha"])
df["diferencia"] = (0.556 * df["pesados"] + 0.444 * df["ligeros"] - df["total"]).abs()
print("Meses:", len(df), "|", df["fecha"].min().strftime("%Y-%m"), "a", df["fecha"].max().strftime("%Y-%m"))
print("Meses faltantes:", len(pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="MS")) - len(df))
print("Diferencia máxima:", round(df["diferencia"].max(), 3))
print(df[df["diferencia"] > 0.3])