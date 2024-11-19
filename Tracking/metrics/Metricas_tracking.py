import cv2
import numpy as np
import time
from pathlib import Path

from boxmot import DeepOCSORT
from boxmot import BoTSORT
from boxmot import HybridSORT
#from boxmot import OCSort
from boxmot import StrongSORT
from boxmot import BYTETracker

from ultralytics.models.yolo.model import YOLO
import os

def xyxy_to_xywh(xyxy, width, height):
    """
    Convierte coordenadas XYXY a XYWH (centro, ancho, alto).

    Parameters:
    - xyxy (list or tuple): Coordenadas XYXY [x1, y1, x2, y2].

    Returns:
    - xywh (list): Coordenadas XYWH [xc, yc, w, h].
    """
    x1, y1, x2, y2 = xyxy
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    return [xc, yc, w, h]

# Rutas

ruta_frames_txt = '.../Escritorio/videos_pesos_GT/GT/video2_v5/obj_train_data'
num_frames = len(os.listdir(ruta_frames_txt))

# Video path
video_path = '.../Escritorio/videos_pesos_GT/Videos_orig/video_2.mp4'

# Detector path
detector_path = '.../Escritorio/videos_pesos_GT/Weights/v5/yolov8_x_1280_v5.pt'

filas = 21  # Bbox existentes
columnas = 5  # Id, centroide (x, y) ancho alto bbox (w, h)
profundidad = num_frames  # Datos por cada frame para adaptar el volumen al video concreto
duracion_promedio = 0 # Para el cálculo de FPS promedio. Mejor explicado posteriormente
id_maximo_asignado = 0 # Para contar cual es número máximo de id que asigna el tracker

# Dimensiones imagenes DeepFarm
real_width = 1920
real_height = 3452

ground_truth_vol = np.zeros((filas, columnas, profundidad), dtype='float32')  # Volumen matricial referencia. CVAT Raúl.

for p in range(profundidad):
    # Generar el nombre del archivo para leer
    nombre_archivo = '/frame_' + str(p).zfill(6) + '.txt'
    ruta_archivo = ruta_frames_txt + nombre_archivo

    with open(ruta_archivo) as archivo:
        for fila_idx, linea in enumerate(archivo):
            valores = list(map(float, linea.strip().split()))
            ground_truth_vol[fila_idx, :, p] = valores

ground_truth_vol[:, 0, :] += 1  # Pasa los id al rango 1-21.

continuidad_trayectorias = np.zeros((filas, profundidad),
                                    dtype='int8')  # Se almacena para cada instante de tiempo (frame)
                                                   # de cada id si se ha trackeado bien.
                                                   # Y si en algún momento vuelve a encontrar un id perdido se ve.

resultado_trayectorias = np.zeros((filas, 4),
                                  dtype='int16')  # Se almacenan los resultados para cada frame de la continuidad del
                                                  # tracklet. PÉRDIDAS DE ID (nº de pasos de 1 a 0), RECUPERACIONES DE ID
                                                  # (nº de pasos de 0 a 1), FRAMES CON ID PERDIDO (frames a 0),
                                                  # MISMO ID INICIO/FIN (0 si se pierde y
                                                  # no recupera 1 si se mantiene o se recupera)

vid = cv2.VideoCapture(video_path)
color = (0, 0, 255)  # BGR
thickness = 2
fontscale: float = 2

# Selección del tracker
#tracker = DeepOCSORT(
#     model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
#    model_weights = Path('.../Escritorio/ultralytics_github/Tracking_deep_farm/log/osnet_x1_0_DSM-RHM/model/osnet_x1_0_DeeFarm.pt'),
#    model_weights=Path('.../Escritorio/ultralytics_github/Tracking_deep_farm/log/osnet_x1_0_050124/model/osnet_x1_0_DeepFarm050124.pt'),
#    device='cuda:0',
#     fp16=True
#)

tracker = BoTSORT(
     #model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    # LKa arquitectura tiene que ir en el nombre del archivo

    model_weights=Path('.../Escritorio/ultralytics_github/Tracking_deep_farm/log/osnet_x1_0_050124/model/osnet_x1_0.pt'),
    device='cuda:0',
     fp16=True,
)


#tracker = BYTETracker(

#)

#tracker = StrongSORT(
#     model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
#    device='cuda:0',
#     fp16=True,
#)

#tracker = OCSort(
#     model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
#    device='cuda:0',
#     fp16=True,
#)

#tracker = HybridSORT(
#     reid_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
#    device='cuda:0',
#    det_thresh = 0.8,
#    half = 2
#)


# Selección del modelo

model = YOLO(detector_path)
a = 0
num_frame = 0 # Identifica el frame para el print por consola. no tiene ninguna otra labor lógica
while True:

    ini_time = time.time() # Se registra el instante del inicio de la iteración

    ret, im = vid.read()

    if not ret:
        print('-------------------------------------')
        print('----------- Fin del video -----------')
        print('-------------------------------------')
        print('')
        break

    # substitute by your object detector, input to tracker has to be N X (x, y, x, y, conf, cls)
    l_results = model(im)

    num_bboxes = l_results[0].boxes.xyxy.size(0)
    results = np.zeros((num_bboxes, 6), dtype = 'float32')

    # Logística de cuda, tensores, cpu, etc
    boxes_gpu = l_results[0].boxes.xyxy
    cls_gpu = l_results[0].boxes.cls
    conf_gpu = l_results[0].boxes.conf

    boxes_cpu = boxes_gpu.cpu()
    cls_cpu = cls_gpu.cpu()
    conf_gpu = conf_gpu.cpu()

    res_boxes = boxes_cpu.numpy()
    res_cls = cls_cpu.numpy()
    res_conf = conf_gpu.numpy()

    # Generación de results
    for b in range(num_bboxes):
        results[b, 0:4] = res_boxes[b]
        results[b, 4] = res_conf[b]
        results[b, 5] = res_cls[b]

    # tracking de results
    tracks = tracker.update(results, im) # --> (x, y, x, y, id, conf, cls, ind)

    xyxys = tracks[:, 0:4].astype('int') # float64 to int
    ids = tracks[:, 4].astype('int') # float64 to int
    confs = tracks[:, 5]
    clss = tracks[:, 6].astype('int') # float64 to int
    inds = tracks[:, 7].astype('int') # float64 to int

    fin_time = time.time() # Se registra el final de la iteración. Solo se considera el tiempo
                           # que transcurre entre detección y tracking, no el de generar trayectorias.

    # Generación de trayectorias

    for b in range(ids.shape[0]): # Recorre las cajas de cada resultado

        if id_maximo_asignado < ids[b]: # Actualización del id máximo asignado
            id_maximo_asignado = ids[b]

        if ids[b] in ground_truth_vol[:, 0, a]: # Si el id de la caja de turno está en el GT de ese frame entra
            id_gt_array = np.argwhere(ground_truth_vol[:, 0, a].astype(int) == ids[b]) # Obtiene la posición del id de la caja detectada en el array GT
            id_gt = id_gt_array[0]

            x_gt = ground_truth_vol[id_gt, 1, a] * real_width  # Se obtienen las coordenadas x e y del centroide GT y se multiplica por la magnitud real
            y_gt = ground_truth_vol[id_gt, 2, a] * real_height # para trabajar en distancias no normalizadas. Se podría trabajar con distancias normalizadas también.

            [x_pred, y_pred, w_pred, h_pred] = xyxy_to_xywh(xyxys[b], real_width, real_height)

            dist_euc = np.sqrt((x_gt - x_pred)**2 + (y_gt - y_pred)**2) # Cálculo de la distancia euclidea entre los centroides

            if dist_euc < 60:
                continuidad_trayectorias[ids[b] - 1, a] = 1 # esto se hace así porque ids empieza por el id mayor [22, 21, 20 ... ]
    print(id_maximo_asignado)                                                          # y la b se incrementa aunque no procese por no ser un id dentro del GT
    a += 1
    # Traigo este cálculo aquí para poder aprovechar la variable a en el cálculo de la duración promedio.
    duracion = fin_time - ini_time  # Determinación del lapso transcurrido
    duracion_promedio = (duracion_promedio * a + duracion) / (a + 1) # calculo de media con el resultado de la nueva iteración

    # Fin generación de trayectorias

    # print bboxes with their associated id, cls and conf
    if tracks.shape[0] != 0:
        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
            im = cv2.rectangle(
                im,
                (xyxy[0], xyxy[1]),
                (xyxy[2], xyxy[3]),
                color,
                thickness
            )
            cv2.putText(
                im,
                f'id: {id}',
#                f'id: {id}, conf: {conf}, c: {cls}',
                (xyxy[0], xyxy[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness
            )

    # Determinación del sttiching
    #color_linea = (0, 255, 0)
    #cv2.line(im, (0, 1050), (1920-1, 1050), color_linea, thickness = 10)
    #cv2.line(im, (0, 1700), (1920 - 1, 1700), color_linea, thickness=10)
    #cv2.line(im, (0, 2350), (1920 - 1, 2350), color_linea, thickness=10)
    # Fin determinación de sttiching

    # show image with bboxes, ids, classes and confidences
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', im)
    print('Frame ' + str(num_frame))
    num_frame += 1

    # break on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

# Cálculo de métricas
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
FPS_promedio = 1 / duracion_promedio

print('------------ RESULTADOS -------------')
print('IDF1 = ' + str(IDF1))
print('-------------------------------------')
print('SWID absoluto = ' + str(sum(resultado_trayectorias[:,0]))) # Cuantas transiciones de 1 a 0 existen en total
print('Recuperaciones absoluto = ' + str(sum(resultado_trayectorias[:,1]))) # Transiciones de 1 a 0 totales dividido entre el total detecciones
print('-------------------------------------')
print('AttMetric (%) = ' + str(100*sum(resultado_trayectorias[:,3])/filas) + '%') # Porcentaje de terneros que tienen el mismo id al principio y al final del video
print('-------------------------------------')
print(' TP --> ' + str(TP))
print(' Errores --> ' + str(errores))
print('-------------------------------------')
print('FPS promedio del tracker + detector (FPS)= ' + str(FPS_promedio))
print('El id máximo asignado ha sido: ' + str(id_maximo_asignado))

# Se almacenan los resultados para cada frame de la continuidad del
# tracklet. [0] PÉRDIDAS DE ID (nº de pasos de 1 a 0), [1] RECUPERACIONES DE ID
# (nº de pasos de 0 a 1), [2] FRAMES CON ID PERDIDO (frames a 0),
# [3] MISMO ID INICIO/FIN (0 si se pierde y no recupera 1 si se mantiene o se recupera)