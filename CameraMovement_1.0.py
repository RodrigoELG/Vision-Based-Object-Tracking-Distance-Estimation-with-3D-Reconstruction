import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Función para detectar los bordes y encontrar los puntos de intersección
def find_intersections(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    intersections = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1, line2 = lines[i][0], lines[j][0]
                intersection = compute_intersection(line1, line2)
                if intersection:
                    intersections.append(intersection)
                    cv2.circle(img, intersection, 5, (0, 0, 255), -1)  
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

# Carga y calibración de la cámara desde una imagen
def calibrate_camera_from_image(image_path):
    img = cv2.imread(image_path)
    assert img is not None, "Error: la imagen no pudo ser cargada."
    height, width = img.shape[:2]
    aspect_ratio = width / height
    new_width = 500
    new_height = int(new_width / aspect_ratio)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    img_with_intersections, intersections = find_intersections(img)

    # Almacenamiento de las intersecciones detectadas (para uso futuro)
    img_name = os.path.basename(image_path)
    puntos_ref = {img_name: intersections}
    with open('puntos_referencia.json', 'w') as fp:
        json.dump(puntos_ref, fp)

    puntos_imagen = []
    puntos_objeto = []
    credencial_ancho = 30 
    credencial_alto = 30  

    for img_name, puntos in puntos_ref.items():
        for punto in puntos:
            puntos_imagen.append([punto])
            x, y = punto
            puntos_objeto.append([(float(x / img.shape[1]) * credencial_ancho, float(y / img.shape[0]) * credencial_alto, 0.0)])

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

    # Calcular la longitud focal en cm
    tamano_pixel_cm = 0.000012  # Tamaño del pixel en cm (12 micrómetros convertido a cm)
    focal_length_width = mtx[0][0] * tamano_pixel_cm
    focal_length_height = mtx[1][1] * tamano_pixel_cm 

    print("\nFocal width (cm):")
    print(focal_length_width)
    print("\nFocal height (cm):")
    print(focal_length_height)

    # Mostrar imagen con intersecciones detectadas
    cv2.imshow('Intersections', img_with_intersections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return mtx, dist

# Función principal para rastrear la trayectoria de la cámara
def track_camera_trajectory(video_path, calib_mtx, calib_dist):
    # Inicialización de la captura de video
    video_cap = cv2.VideoCapture(video_path)
    ret, last_frame = video_cap.read()

    # Inicialización de variables para la trayectoria de la cámara
    t = np.zeros((3, 1))  # Vector de traslación acumulada
    camera_translation = []  # Lista para guardar las traslaciones
    frame_count = 0

    # Inicialización de ORB y BFMatcher
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher()

    # Configuración de la visualización de la trayectoria en vivo
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()  # Modo interactivo

    # Bucle principal para procesar cada cuadro del video
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        query_img = frame_bw.copy()
        train_img = last_frame.copy()

        # Detectar y describir características en el cuadro actual y anterior
        queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img, None)
        trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img, None)

        # Emparejar las características entre cuadros
        matches = matcher.match(queryDescriptors, trainDescriptors)
        matches = sorted(matches, key=lambda x: x.distance)[:100] #100 

        # Convertir los puntos emparejados en arrays numpy
        query_idx = [m.queryIdx for m in matches]
        train_idx = [m.trainIdx for m in matches]
        query_points = cv2.KeyPoint.convert(queryKeypoints, query_idx)
        train_points = cv2.KeyPoint.convert(trainKeypoints, train_idx)

        # Calcular la matriz esencial y recuperar la pose de la cámara
        E, mask = cv2.findEssentialMat(query_points, train_points, calib_mtx)
        _, R, T, mask = cv2.recoverPose(E, query_points, train_points, calib_mtx)

        # Acumular la traslación para rastrear la trayectoria de la cámara
        t += T
        camera_translation.append(t.copy())

        frame_count += 1
        if frame_count % 10 == 0:
            # Actualización de la gráfica en vivo
            pos = np.array(camera_translation)
            ax.clear()
            ax.plot(pos[:, 0, 0], pos[:, 1, 0], pos[:, 2, 0], label='Camera Trajectory')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.draw()
            plt.pause(0.01)

        # Visualización de las coincidencias
        img_matches = cv2.drawMatches(query_img, queryKeypoints, train_img, trainKeypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('ORB Matches', img_matches)

        if cv2.waitKey(1) == ord(' '):
            break

        last_frame = frame_bw

    # Liberar recursos
    video_cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    image_path = r'C:\Users\rodri\MR5 TrackingPatos\PatronLoza\Captura de pantalla 2024-05-15 222749.png'
    video_path = r'C:\Users\rodri\MR5 TrackingPatos\DuckVideo.mp4'
    
    # Calibrar la cámara desde una imagen
    calib_mtx, calib_dist = calibrate_camera_from_image(image_path)
    
    # Rastrear la trayectoria de la cámara usando el video
    track_camera_trajectory(video_path, calib_mtx, calib_dist)
