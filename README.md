# Predicción del Flujo Vehicular en Peajes (INEI)

Prototipo en Python y Streamlit para el análisis y pronóstico mensual del Índice Nacional del Flujo Vehicular en peajes (base 2007 = 100), alineado con los objetivos ODS 9 y ODS 11.

## Estructura del proyecto
* `data/indice_flujo_vehicular.csv`: Serie histórica mensual (103 meses).
* `src/verificar_datos.py`: Validación de integridad y consistencia de las ponderaciones del INEI.
* `src/modelos.py`: Implementación de los 6 modelos de pronóstico (Media móvil, Estacional ingenuo, Holt-Winters, SARIMA, Random Forest y Red neuronal multicapa).
* `src/entrenar.py`: Evaluación cronológica y generación de métricas en CSV.
* `src/app.py`: Interfaz web interactiva desarrollada en Streamlit.
* `results/`: Tablas de métricas, predicciones y variación interanual generadas.

## Instrucciones de ejecución

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verificar la base de datos:**
   ```bash
   python src/verificar_datos.py
   ```

3. **Entrenar y evaluar los modelos:**
   ```bash
   python src/entrenar.py --serie total
   ```

4. **Ejecutar la aplicación web:**
   ```bash
   streamlit run src/app.py
   ```
