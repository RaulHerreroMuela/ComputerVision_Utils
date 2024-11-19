from ultralytics.models.yolo.model import YOLO
import numpy as np
import os

# Rutas
# Ground Truth
ruta_frames_txt = '.../Escritorio/.../obj_train_data'
num_frames = len(os.listdir(ruta_frames_txt))

# Video path
video_path = '..../Escritorio/videos_pesos_GT/Videos_orig/video_1.mp4'

# Tracker path
tracker_path = '.../Escritorio/Tracking/BOX_MOT/yolo_tracking/boxmot/configs/bytetrack.yaml'

# Detector path
detector_path = '.../Escritorio/videos_pesos_GT/Weights/v2/yolov8_x_1280_.pt'

filas = 21 # Bbox existentes
columnas = 5 # Id, centroide (x, y) ancho alto bbox (w, h)
profundidad = num_frames # Datos por cada frame para adaptar el volumen al video concreto

# Dimensiones imagenes DeepFarm
real_width = 1920
real_height = 3452

ground_truth_vol = np.zeros((filas, columnas, profundidad), dtype = 'float32')   # Volumen matricial referencia. CVAT Raúl.

for p in range(profundidad):
    # Generar el nombre del archivo para leer
    nombre_archivo = '/frame_' + str(p).zfill(6) + '.txt'
    ruta_archivo = ruta_frames_txt + nombre_archivo

    with open(ruta_archivo) as archivo:
        for fila_idx, linea in enumerate(archivo):
            
            valores = list(map(float, linea.strip().split()))
            ground_truth_vol[fila_idx, :, p] = valores

ground_truth_vol[:,0,:] += 1 # Pasa los id al rango 1-21.

continuidad_trayectorias = np.zeros((filas, profundidad), dtype = 'int8') # Se almacena para cada instante de tiempo (frame)
                                                                                # de cada id si se ha trackeado bien.
                                                                                # Y si en algún momento vuelve a encontrar un id perdido se ve.

resultado_trayectorias = np.zeros((filas, 4), dtype = 'int16') # Se almacenan los resultados para cada frame de la continuidad del
                                                                     # tracklet. PÉRDIDAS DE ID (nº de pasos de 1 a 0), RECUPERACIONES DE ID
                                                                     # (nº de pasos de 0 a 1), FRAMES CON ID PERDIDO (frames a 0),
                                                                     # MISMO ID INICIO/FIN (0 si se pierde y
                                                                     # no recupera 1 si se mantiene o se recupera)

# Letura del modelo y llamada al tracker
model = YOLO(detector_path)

results = model.track(source = video_path,
                      stream = True,
                      agnostic_nms = True,
                      iou = 0.5, 
                      show = True,
                      save = False,
                      persist = True,
                      tracker = tracker_path)

TP = 0 # Variable que almacena las detecciones correctas
TP_aux = 0 # Contador auxiliar para corregir el número de errores cuando hay menos de 21 detecciones.
error = 0 # Variable que almacena los errores de detección
error_aux = 0 # Idem TP_aux
a = 0 # va a recorrer la profundidad del volumen

for r in results: # Recorre los resultados de cada frame
    for b in r.boxes: # Recorre las cajas de cada resultado
        if int(b.id) in ground_truth_vol[:, 0, a]: # Si el id de la caja de turno está en el GT de ese frame entra
            id_gt_array = np.argwhere(ground_truth_vol[:, 0, a].astype(int) == int(b.id)).flatten() # Obtiene la posición del id de la caja detectada en el array GT
            id_gt = id_gt_array[0]

            x_gt = ground_truth_vol[id_gt, 1, a] * real_width  # Se obtienen las coordenadas x e y del centroide GT y se multiplica por la magnitud real
            y_gt = ground_truth_vol[id_gt, 2, a] * real_height # para trabajar en distancias no normalizadas. Se podría trabajar con distancias normalizadas también.

            cent_det = np.squeeze(b.xywhn.numpy()) # Se obtienen las coordenadas x e y del centroido del detector y se multiplica por la magnitud real
            x_pred = cent_det[0] * real_width      # para trabajar en distancias no normalizadas. Se podría trabajar con distancias normalizadas también.
            y_pred = cent_det[1] * real_height

            dist_euc = np.sqrt((x_gt - x_pred)**2 + (y_gt - y_pred)**2) # Cálculo de la distancia euclidea entre los centroides

            if dist_euc < 41:
                continuidad_trayectorias[int(b.id) - 1, a] = 1
    a += 1


for fil in range(filas):

    resultado_trayectorias[fil, 2] = np.count_nonzero(continuidad_trayectorias[fil,:] == 0) # Conteo de 0

    if ((continuidad_trayectorias[fil,0] == continuidad_trayectorias[fil, profundidad - 1]) # Compara el id de cada tracklet en el
            and (continuidad_trayectorias[fil,0] == 1)):                                    # frame 0 y último. Si son iguales e igual a 1
        resultado_trayectorias[fil, 3] = 1                                                  # se pone el indicador a 1.

    for frame in range(profundidad - 1): # Aquí se recorre cada trayectoria de tracklet para ver la cantidad de transiciones ya sea de 1 a 0 ó de 0 a 1.

        if continuidad_trayectorias[fil, frame] != continuidad_trayectorias[fil, frame + 1]: # Si el valor del frame siguiente y actual no coinciden se mira a qué contador
            if ((continuidad_trayectorias[fil, frame] == 1)                                  # hay que asignar la perdida o el encuentro.
                and (continuidad_trayectorias[fil, frame + 1] == 0)):                        # Pasar de 1 a 0 implica una pérdida, así que incrementamos el contador de pérdida.
                resultado_trayectorias[fil,0] += 1

            if ((continuidad_trayectorias[fil, frame] == 0)         # Pasar de 0 a 1 implica un encuentro y por tanto incrementamos el contador de encuentros.
                and (continuidad_trayectorias[fil, frame + 1] == 1)):
                resultado_trayectorias[fil, 1] += 1

errores = sum(resultado_trayectorias[:,2])
TP = profundidad * 21 - errores
IDF1 = (2*TP)/(2*TP + errores)

print('------------ RESULTADOS -------------')
print('IDF1 = ' + str(IDF1))
print('-------------------------------------')
print('SWID absoluto = ' + str(sum(resultado_trayectorias[:,0]))) # Cuantas transiciones de 1 a 0 existen en total
print('Recuperaciones absoluto = ' + str(sum(resultado_trayectorias[:,1]))) # Transiciones de 1 a 0 totales dividido entre el total detecciones
print('-------------------------------------')
print('AttMetric (%) = ' + str(100*sum(resultado_trayectorias[:,3])/filas) + '%') # Porcentaje de terneros que tienen el mismo id al principio y al final del video
print('-------------------------------------')
# Se almacenan los resultados para cada frame de la continuidad del
# tracklet. [0] PÉRDIDAS DE ID (nº de pasos de 1 a 0), [1] RECUPERACIONES DE ID
# (nº de pasos de 0 a 1), [2] FRAMES CON ID PERDIDO (frames a 0),
# [3] MISMO ID INICIO/FIN (0 si se pierde y no recupera 1 si se mantiene o se recupera)