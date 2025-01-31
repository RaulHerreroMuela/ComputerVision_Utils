import numpy as np
import cv2
import os
import jupyter_notebooks.utils.aux_functions as aux

# Funciones
def draw_polygon(event, x, y, flags, param):
    global points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Función para dibujar líneas sobre la imagen
def draw_lines(event, x, y, flags, param):
    global drawing, ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(mask, (ix, iy), (x, y), (255, 255, 255), thickness=5)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

imagen_original_path = aux.ask_file(target = 'file', title = 'Seleccione imagen a procesar')
imagen_original = cv2.imread(imagen_original_path)
height, width, channels = imagen_original.shape

path_save_mask = aux.ask_file(target = 'folder', title = 'Seleccione ubicacion para guardar la mascara')

# Dibujar roi sobre la imagen
# Cargar la imagen
mask = np.zeros_like(imagen_original[:, :, 0])

# Crear una ventana para la imagen
cv2.namedWindow('Imagen', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Imagen', imagen_original.shape[1], imagen_original.shape[0])  # Ajustar tamaño de la ventana a las dimensiones de la imagen
cv2.setMouseCallback('Imagen', draw_polygon)
#cv2.setMouseCallback('Imagen', draw_lines)

drawing = False
points = []

# Loop para dibujar un polígono sobre la imagen
while True:
    clone = imagen_original.copy()
    if len(points) > 1:
        cv2.polylines(clone, np.array([points]), False, (255, 255, 255), thickness=5)

    cv2.imshow('Imagen', clone)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):  # Presionar 'r' para borrar el dibujo
        points = []
        clone = imagen_original.copy()
    if key == 13:  # Presionar 'Enter' para salir del loop
        break

cv2.destroyAllWindows()

# Rellenar el polígono en la máscara
cv2.fillPoly(mask, [np.array(points)], (255, 255, 255))

# Convertir la máscara a binario
binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

# Guardar la máscara binaria
cv2.imwrite(path_save_mask + '/' + 'mask_agua_chiva_pozo_bombeo.jpg', binary_mask)

# Mostrar la máscara binaria
cv2.imshow('Máscara binaria', binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()