from pathlib import Path

import cv2
from deepface import DeepFace


CARPETA_CONOCIDOS = Path("conocidos")
CARPETA_PRUEBAS = Path("pruebas")
CARPETA_RESULTADOS= Path("resultados")

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODELO_RECONOCIMIENTO = "Facenet"
DETECTOR_BACKEND = "opencv"

UMBRAL_COINCIDENCIA = 0.40


def cargar_personas_conocidas():
    """
    Lee las imágenes de la carpeta 'conocidos'.

    Cada archivo debe llamarse como la persona, por ejemplo:
    conocidos/Ana.png
    conocidos/Carlos.png
    """
    personas = []

    if not CARPETA_CONOCIDOS.exists():
        raise FileNotFoundError(f"No existe la carpeta '{CARPETA_CONOCIDOS}'.")

    for ruta_imagen in sorted(CARPETA_CONOCIDOS.iterdir()):
        if ruta_imagen.suffix.lower() != ".png":
            continue

        nombre = ruta_imagen.stem
        personas.append(
            {
                "nombre": nombre,
                "ruta": ruta_imagen,
            }
        )

    return personas


def obtener_imagenes_de_prueba():
    """
    Devuelve todas las imágenes válidas dentro de la carpeta 'pruebas'.
    """
    if not CARPETA_PRUEBAS.exists():
        raise FileNotFoundError(f"No existe la carpeta '{CARPETA_PRUEBAS}'.")

    imagenes = [
        ruta
        for ruta in sorted(CARPETA_PRUEBAS.iterdir())
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN
    ]

    return imagenes


def detectar_caras(ruta_imagen):
    """
    Detecta las caras de una imagen usando DeepFace.

    Devuelve una lista de diccionarios con información de cada cara detectada.
    """
    return DeepFace.extract_faces(
        img_path=str(ruta_imagen),
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
        align=True,
    )


def comparar_conocidos(ruta_imagen_prueba, area_cara, personas_conocidas):
    """
    Compara una cara detectada en una imagen de prueba contra todas las personas conocidas.

    Devuelve el nombre de la persona si hay coincidencia.
    Si no hay coincidencia, devuelve None.
    """
    mejor_nombre = None
    mejor_distancia = None

    x = area_cara["x"]
    y = area_cara["y"]
    ancho = area_cara["w"]
    alto = area_cara["h"]

    imagen = cv2.imread(str(ruta_imagen_prueba))

    if imagen is None:
        return None

    cara = imagen[y:y + alto, x:x + ancho]

    if cara.size == 0:
        return None

    for persona in personas_conocidas:
        try:
            resultado = DeepFace.verify(
                img1_path=cara,
                img2_path=str(persona["ruta"]),
                model_name=MODELO_RECONOCIMIENTO,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True,
            )

            distancia = resultado["distance"]

            if mejor_distancia is None or distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_nombre = persona["nombre"]

        except Exception as error:
            print(f"No se pudo comparar con {persona['nombre']}: {error}")

    if mejor_distancia is not None and mejor_distancia <= UMBRAL_COINCIDENCIA:
        return mejor_nombre

    return None


def dibujar_etiqueta(imagen, texto, x, y, color):
    """
    Dibuja una etiqueta con fondo sobre la imagen.
    """
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.7
    grosor = 2

    margen = 6
    ancho_texto, alto_texto = cv2.getTextSize(texto, fuente, escala, grosor)[0]

    y_texto = max(y - 10, alto_texto + margen * 2)

    cv2.rectangle(
        imagen,
        (x, y_texto - alto_texto - margen * 2),
        (x + ancho_texto + margen * 2, y_texto),
        color,
        cv2.FILLED,
    )

    cv2.putText(
        imagen,
        texto,
        (x + margen, y_texto - margen),
        fuente,
        escala,
        (255, 255, 255),
        grosor,
        cv2.LINE_AA,
    )


def procesar_imagen(ruta_imagen, personas_conocidas):
    """
    Procesa una imagen de prueba:
    - detecta caras,
    - compara cada cara con las conocidas,
    - dibuja rectángulos y etiquetas.

    Devuelve el conjunto de personas conocidas encontradas en la imagen.
    """
    imagen = cv2.imread(str(ruta_imagen))

    if imagen is None:
        print(f"No se pudo leer la imagen: {ruta_imagen}")
        return set()

    caras = detectar_caras(ruta_imagen)
    personas_encontradas = set()

    for cara_detectada in caras:
        area = cara_detectada.get("facial_area")

        if not area:
            continue

        x = area["x"]
        y = area["y"]
        ancho = area["w"]
        alto = area["h"]

        nombre = comparar_conocidos(ruta_imagen, area, personas_conocidas)

        if nombre is None:
            etiqueta = "Desconocido"
            color = (0, 0, 255)
        else:
            etiqueta = nombre
            color = (0, 255, 0)
            personas_encontradas.add(nombre)

        cv2.rectangle(
            imagen,
            (x, y),
            (x + ancho, y + alto),
            color,
            2,
        )

        dibujar_etiqueta(imagen, etiqueta, x, y, color)

        ruta_resultado=CARPETA_RESULTADOS / ruta_imagen.name
        cv2.imwrite(str(ruta_resultado), imagen)

    cv2.imshow(f"Resultado - {ruta_imagen.name}", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return personas_encontradas


def main():
    personas_conocidas = cargar_personas_conocidas()
    imagenes_prueba = obtener_imagenes_de_prueba()

    if not personas_conocidas:
        print("No se encontraron personas conocidas en la carpeta 'conocidos'.")
        return

    if not imagenes_prueba:
        print("No se encontraron imágenes para analizar en la carpeta 'pruebas'.")
        return

    personas_distintas_encontradas = set()

    print(f"Personas conocidas cargadas: {len(personas_conocidas)}")
    print(f"Fotos de prueba encontradas: {len(imagenes_prueba)}")
    print()

    for indice, ruta_imagen in enumerate(imagenes_prueba, start=1):
        print(f"Analizando foto {indice}/{len(imagenes_prueba)}: {ruta_imagen.name}")

        encontradas = procesar_imagen(ruta_imagen, personas_conocidas)
        personas_distintas_encontradas.update(encontradas)

    print()
    print("Resumen final")
    print("-------------")
    print(f"Fotos analizadas en total: {len(imagenes_prueba)}")
    print(
        "Personas conocidas distintas encontradas: "
        f"{len(personas_distintas_encontradas)}"
    )

    if personas_distintas_encontradas:
        print(
            "Nombres encontrados: "
            + ", ".join(sorted(personas_distintas_encontradas))
        )


if __name__ == "__main__":
    main()