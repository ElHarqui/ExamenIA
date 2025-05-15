import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------- Cargar imágenes ----------
def cargar_imagenes(ruta_base, tamaño=(64, 64)):
    X = []
    y = []
    etiquetas = {'laptop': 0, 'celular': 1, 'tablet': 2}

    for etiqueta, valor in etiquetas.items():
        carpeta = os.path.join(ruta_base, etiqueta)
        if not os.path.exists(carpeta):
            print(f"⚠️ Carpeta no encontrada: {carpeta}")
            continue
        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                ruta_img = os.path.join(carpeta, archivo)
                try:
                    img = Image.open(ruta_img).convert('L')  # blanco y negro
                    img = img.resize(tamaño)
                    vector = np.array(img).flatten()
                    X.append(vector)
                    y.append(valor)
                except UnidentifiedImageError:
                    print(f"❌ Imagen no válida (no identificable): {ruta_img}")
                except Exception as e:
                    print(f"⚠️ Error procesando {ruta_img}: {e}")
    return np.array(X), np.array(y)

# ---------- Predecir imagen nueva ----------
def predecir_imagen(modelo, ruta_img):
    etiquetas_inversas = {0: "Laptop 💻", 1: "Celular 📱", 2: "Tablet 📲"}
    try:
        img = Image.open(ruta_img).convert('L').resize((64, 64))
        vector = np.array(img).flatten().reshape(1, -1)
        pred = modelo.predict(vector)
        return etiquetas_inversas.get(pred[0], "Desconocido")
    except Exception as e:
        return f"⚠️ Error al procesar la imagen: {e}"

# ---------- Main ----------
if __name__ == '__main__':
    modelo_path = "modelo_dispositivos.pkl"

    # Si existe el modelo, cargarlo
    if os.path.exists(modelo_path):
        print("📦 Cargando modelo entrenado...")
        modelo = joblib.load(modelo_path)
    else:
        print("⚙️ Entrenando nuevo modelo...")
        X, y = cargar_imagenes("dataset")

        if len(X) < 6:
            print("⚠️ Necesitas al menos 2 imágenes por clase para entrenar.")
            exit()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        print("\n=== Resultados ===")
        print("Precisión:", accuracy_score(y_test, y_pred))s
        print(classification_report(y_test, y_pred, zero_division=1))

        joblib.dump(modelo, modelo_path)
        print(f"✅ Modelo guardado en {modelo_path}")

    # Predecir una imagen específica
    nueva_imagen = "dataset/test.jpg"  # <- cambia si necesitas
    if os.path.exists(nueva_imagen):
        resultado = predecir_imagen(modelo, nueva_imagen)
        print(f"\nResultado de la imagen '{nueva_imagen}': {resultado}")
    else:
        print(f"\n⚠️ No se encontró la imagen: {nueva_imagen}")
