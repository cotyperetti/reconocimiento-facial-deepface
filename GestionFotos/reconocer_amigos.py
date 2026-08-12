"""
reconocer_amigos.py

Organiza fotos por reconocimiento facial usando deepface y opencv.

- Lee las caras de referencia desde la carpeta "conocidos" (una foto por
  persona, con el nombre de archivo = nombre de la persona).
- Recorre cada foto de la carpeta "pruebas", detecta todas las caras que
  aparecen y las compara contra las de "conocidos".
- Dibuja un rectángulo verde + nombre sobre las caras reconocidas, y un
  rectángulo rojo + "Desconocido" sobre las que no coinciden con nadie.
- Muestra cada foto procesada en una ventana (se avanza con cualquier tecla).
- Al final imprime un resumen: fotos analizadas y personas conocidas
  distintas encontradas en total.

Requisitos: pip install deepface opencv-python
"""

import os

import cv2
from deepface import DeepFace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONOCIDOS_DIR = os.path.join(BASE_DIR, "conocidos")
PRUEBAS_DIR = os.path.join(BASE_DIR, "pruebas")

MODEL_NAME = "VGG-Face"       # modelo de deepface para embeddings
DETECTOR_BACKEND = "opencv"   # backend de detección de caras


def cargar_conocidos(carpeta):
    """Devuelve una lista de (nombre_persona, ruta_foto) por cada archivo
    .png dentro de la carpeta de conocidos."""
    conocidos = []
    for archivo in sorted(os.listdir(carpeta)):
        if archivo.lower().endswith(".png"):
            nombre = os.path.splitext(archivo)[0]
            ruta = os.path.join(carpeta, archivo)
            conocidos.append((nombre, ruta))
    return conocidos


def detectar_caras(ruta_imagen):
    """Detecta todas las caras de una imagen y devuelve una lista de
    regiones (x, y, w, h) usando el mismo backend que usaremos para
    verificar identidades."""
    try:
        caras = DeepFace.extract_faces(
            img_path=ruta_imagen,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
    except Exception as e:
        print(f"  ! Error detectando caras en {os.path.basename(ruta_imagen)}: {e}")
        return []

    regiones = []
    for cara in caras:
        area = cara.get("facial_area", {})
        x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)
        confianza = cara.get("confidence", 0)
        # descartar detecciones vacías/espurias (p. ej. cuando no hay cara real)
        if w > 0 and h > 0 and confianza > 0:
            regiones.append((x, y, w, h))
    return regiones


def identificar_cara(ruta_imagen, region, conocidos):
    """Recorta la región de la cara (x, y, w, h) y la compara contra cada
    persona conocida. Devuelve el nombre de la primera coincidencia o
    None si no coincide con nadie."""
    x, y, w, h = region
    imagen = cv2.imread(ruta_imagen)
    recorte = imagen[y:y + h, x:x + w]

    if recorte.size == 0:
        return None

    for nombre, ruta_conocido in conocidos:
        try:
            resultado = DeepFace.verify(
                img1_path=recorte,
                img2_path=ruta_conocido,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
            )
            if resultado.get("verified"):
                return nombre
        except Exception as e:
            print(f"  ! Error comparando con {nombre}: {e}")
            continue

    return None


def procesar_foto(ruta_imagen, conocidos, personas_encontradas):
    """Procesa una foto de la carpeta pruebas: detecta caras, las
    identifica, dibuja los rectángulos y muestra el resultado."""
    nombre_archivo = os.path.basename(ruta_imagen)
    print(f"Procesando {nombre_archivo}...")

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"  ! No se pudo leer la imagen {nombre_archivo}")
        return

    regiones = detectar_caras(ruta_imagen)
    print(f"  {len(regiones)} cara(s) detectada(s)")

    for region in regiones:
        x, y, w, h = region
        nombre = identificar_cara(ruta_imagen, region, conocidos)

        if nombre:
            color = (0, 255, 0)  # verde (BGR)
            etiqueta = nombre
            personas_encontradas.add(nombre)
        else:
            color = (0, 0, 255)  # rojo (BGR)
            etiqueta = "Desconocido"

        cv2.rectangle(imagen, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            imagen,
            etiqueta,
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    cv2.imshow("Reconocimiento facial - presiona una tecla para continuar", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    conocidos = cargar_conocidos(CONOCIDOS_DIR)
    if not conocidos:
        print("No se encontraron fotos en la carpeta 'conocidos'.")
        return

    print(f"Personas conocidas cargadas: {', '.join(n for n, _ in conocidos)}")

    fotos_prueba = [
        os.path.join(PRUEBAS_DIR, archivo)
        for archivo in sorted(os.listdir(PRUEBAS_DIR))
        if archivo.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    personas_encontradas = set()

    for ruta_imagen in fotos_prueba:
        procesar_foto(ruta_imagen, conocidos, personas_encontradas)

    print("\n--- Resumen ---")
    print(f"Fotos analizadas: {len(fotos_prueba)}")
    print(f"Personas conocidas distintas encontradas: {len(personas_encontradas)}")
    if personas_encontradas:
        print(f"  ({', '.join(sorted(personas_encontradas))})")


if __name__ == "__main__":
    main()
