import cv2
import torch
import numpy as np
import os
import json

# Función para detectar los bordes y encontrar los puntos de intersección
def find_intersections(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    intersections = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Dibuja las líneas en amarillo

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1, line2 = lines[i][0], lines[j][0]
                intersection = compute_intersection(line1, line2)
                if intersection:
                    intersections.append(intersection)
                    cv2.circle(img, intersection, 5, (0, 0, 255), -1)  # Dibuja los puntos de intersección en rojo
    return img, intersections

# Función para calcular la intersección de dos líneas
def compute_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return int(px), int(py)

# Función para calcular la distancia utilizando tamaño
def calculate_distance(apparent_size, real_size_width, real_size_height, focal_length_width, focal_length_height):
    if apparent_size[0] == 0 or apparent_size[1] == 0:
        return float('inf')  # Evita divisiones por cero
    distance_width = (real_size_width * focal_length_width) / apparent_size[0]
    distance_height = (real_size_height * focal_length_height) / apparent_size[1]
    distance = (distance_width + distance_height) / 2
    distance /= .37  # Convertir mm a cm
    return distance

# Función para calcular los valores focales
def calculate_focal_lengths(mtx, pixel_size):
    focal_length_width = mtx[0][0] * pixel_size
    focal_length_height = mtx[1][1] * pixel_size
    return focal_length_width, focal_length_height

# Función para ajustar el tamaño del pixel y calcular los valores focales
# al valor focal le damos un rangos de tolerancia dentro de la iteracion de su calculo para encontrar el valor optimo
def adjust_pixel_size_to_focal_range(mtx, target_focal_range=(90, 200), max_iterations=100):
    tolerance = 0.000025  # Incremento/decremento inicial del tamaño del pixel
    pixel_size = 0.00001  # Tamaño inicial del pixel en cm
    iteration_count = 0
    adjusted = False

    while iteration_count < max_iterations:
        focal_length_width, focal_length_height = calculate_focal_lengths(mtx, pixel_size)
        print(f"Current Focal Width: {focal_length_width}, Current Focal Height: {focal_length_height}")

        if target_focal_range[0] <= focal_length_width <= target_focal_range[1] and target_focal_range[0] <= focal_length_height <= target_focal_range[1]:
            adjusted = True
            break

        # Ajustar el tamaño del pixel en función de si está por encima o por debajo del rango deseado
        if focal_length_width > target_focal_range[1] or focal_length_height > target_focal_range[1]:
            pixel_size -= tolerance  # Disminuir si los valores focales son demasiado altos
        else:
            pixel_size += tolerance  # Aumentar si los valores focales son demasiado bajos

        iteration_count += 1

    if not adjusted:
        print("No se pudo ajustar el tamaño del pixel para obtener valores focales dentro del rango especificado.")

    return pixel_size, focal_length_width, focal_length_height

# Carga y calibración de la imagen
def calibrate_camera_from_image(image_path):
    img = cv2.imread(image_path)
    assert img is not None, "Error: la imagen no pudo ser cargada."
    height, width = img.shape[:2]
    aspect_ratio = width / height
    new_width = 500
    new_height = int(new_width / aspect_ratio)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    img_with_intersections, intersections = find_intersections(img)

    img_name = os.path.basename(image_path)
    puntos_ref = {img_name: intersections}
    with open('puntos_referencia.json', 'w') as fp:
        json.dump(puntos_ref, fp)

    puntos_imagen = []
    puntos_objeto = []

    loza_ancho = 30  # Ancho en centímetros
    loza_alto = 30  # Alto en centímetros

    for img_name, puntos in puntos_ref.items():
        for punto in puntos:
            puntos_imagen.append([punto])
            x, y = punto
            puntos_objeto.append([(float(x / img.shape[1]) * loza_ancho, float(y / img.shape[0]) * loza_alto, 0.0)])

    puntos_imagen = np.array(puntos_imagen, dtype=np.float32)
    puntos_objeto_nested = [tuple(punto) for punto in puntos_objeto]
    puntos_objeto = np.array(puntos_objeto_nested, dtype=np.float32)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([puntos_objeto], [puntos_imagen], (img.shape[1], img.shape[0]), None, None)

    print("Matriz de la cámara (parámetros intrínsecos):")
    print(mtx)
    print("\nCoeficientes de distorsión:")
    print(dist)
    print("\nVectores de rotación:")
    print(rvecs)
    print("\nVectores de traslación:")
    print(tvecs)

    # Ajustar el tamaño del pixel para obtener valores focales dentro del rango especificado
    pixel_size, focal_length_width, focal_length_height = adjust_pixel_size_to_focal_range(mtx)
    print("\nAdjusted Pixel Size (cm):")
    print(pixel_size)
    print("\nAdjusted Focal Width (cm):")
    print(focal_length_width)
    print("\nAdjusted Focal Height (cm):")
    print(focal_length_height)

    # Mostrar imagen con intersecciones detectadas
    cv2.imshow('Intersections', img_with_intersections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar la imagen calibrada
    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)

    # Retornar los parámetros de calibración
    return mtx, dist, pixel_size, focal_length_width, focal_length_height

# Función para detección y tracking de patos
def detect_and_track_ducks(video_path, mtx, dist, pixel_size, focal_length_width, focal_length_height, delay):

    duck_height_cm = 8.0
    duck_width_cm = 12.0

    # Cargar el modelo YOLOv5 desde el archivo entrenado
    model_path = r'C:\Users\rodri\yolov5\runs\train\exp5\weights\best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error: No se puede abrir el video."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detección de patos utilizando YOLOv5
        results = model(frame)

        # Extraer coordenadas y etiquetas de los resultados
        detections = results.xyxy[0].cpu().numpy()

        # Encontrar el pato con la mejor confianza de detección 
        best_conf = 0
        best_box = None
        for *box, conf, cls in detections:
            if conf > best_conf:
                best_conf = conf
                best_box = box

        # Enmarcar a los patos detectados
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            label = f'{model.names[int(cls)]} {conf * 100:.1f}%'
            color = (0, 255, 0)  # Verde para todos los patos
            if box == best_box:
                color = (0, 0, 255)  # Rojo para el pato con mayor % de confianza
                # Calcular la distancia utilizando la altura de la caja delimitadora
                apparent_width = x2 - x1
                apparent_height = y2 - y1
                distance_cm = calculate_distance(
                    (apparent_width, apparent_height),
                    duck_width_cm, duck_height_cm,
                    focal_length_width, focal_length_height
                )
                #print(f'Apparent Width: {apparent_width}, Apparent Height: {apparent_height}, Distance: {distance_cm} cm')  
                distance_label = f'Distancia: {distance_cm:.1f} cm'
                cv2.putText(frame, distance_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Mostrar el frame procesado
        cv2.imshow('Ducks Detection', frame)

        if cv2.waitKey(delay) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'C:/Users/rodri/MR5 TrackingPatos/PatronLoza/Captura de pantalla 2024-05-15 222749.png'
    video_path = r'C:\Users\rodri\MR5 TrackingPatos\DuckVideo.mp4'

    # Calibrar la cámara
    mtx, dist, pixel_size, focal_length_width, focal_length_height = calibrate_camera_from_image(image_path)
    
    # Detectar y trackear patos, ajustando el delay por que el video se procesaba muy lento 
    detect_and_track_ducks(video_path, mtx, dist, pixel_size, focal_length_width, focal_length_height, delay=5)