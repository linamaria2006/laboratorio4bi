# Importar librería necesaria para la carga del modelo de ML
from joblib import load

# Clase Model
class Model:

    # Función constructor de la clase Model: Está se encarga de cargar el modelo ML exportado previamente
    def __init__(self,columns):
        self.model = load("assets/modelo.joblib")

    # Función: Está se encargada de realizar las predicciones en base a los datos de entrada
    def make_predictions(self, data):
        result = self.model.predict(data)
        return result