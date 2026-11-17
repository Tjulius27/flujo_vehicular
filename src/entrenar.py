"""Evalúa los seis modelos con validación cronológica y guarda los resultados.
Uso (desde la raíz del proyecto):  python src/entrenar.py --serie total"""
import argparse
import os
from modelos import cargar_serie, limpiar_pandemia, evaluar, variacion_interanual, nivel_de_demanda

p = argparse.ArgumentParser()
p.add_argument("--datos", default="data/indice_flujo_vehicular.csv")
p.add_argument("--serie", default="total", choices=["total", "pesados", "ligeros"])
p.add_argument("--meses_prueba", type=int, default=12)
a = p.parse_args()

original = cargar_serie(a.datos, a.serie)
limpia = limpiar_pandemia(original)
os.makedirs("results", exist_ok=True)

variacion_interanual(original).to_csv(f"results/variacion_interanual_{a.serie}.csv", header=["variacion_%"])
pred, metricas = evaluar(limpia, a.meses_prueba)
pred.to_csv(f"results/predicciones_{a.serie}.csv")
metricas.to_csv(f"results/metricas_{a.serie}.csv", index=False)

_, (u1, u2) = nivel_de_demanda(limpia.iloc[:-a.meses_prueba], limpia)
print(f"Serie: {a.serie} | meses: {len(original)} ({original.index[0]:%Y-%m} a {original.index[-1]:%Y-%m})")
print(f"Entrenamiento: hasta {limpia.index[-a.meses_prueba - 1]:%Y-%m} | prueba: últimos {a.meses_prueba} meses")
print(f"Umbrales de nivel de demanda: u1 = {u1:.1f} | u2 = {u2:.1f}\n")
print(metricas.to_string(index=False))