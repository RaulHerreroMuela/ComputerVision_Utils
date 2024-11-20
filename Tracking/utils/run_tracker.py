import cv2
import numpy as np
import time
from pathlib import Path

from boxmot import DeepOCSORT
from boxmot import BoTSORT
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

# Video path
video_path = '.../Escritorio/videos_pesos_GT/Videos_orig/video_3.mp4'

# Grabación de video
output_video_path = '.../Escritorio/Resultados/videos_1cam/PRUEBA.mp4'
fourcc = cv2.VideoWriter.fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (1920, 1080))

# Detector path
detector_path = '.../Escritorio/videos_pesos_GT/Weights/1cam/yolov8_m_640_v5.pt'

# Dimensiones imagenes
real_width = 1024
real_height = 512


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
    #model_weights = Path('.../Escritorio/ultralytics_github/Tracking_deep_farm/log/osnet_x1_0_DSM-RHM/model/osnet_x1_0_DeeFarm.pt'),
    model_weights=Path('.../Escritorio/ultralytics_github/Tracking_deep_farm/log/osnet_x1_0_050124/model/osnet_x1_0_DeepFarm050124.pt'),
    device='cuda:0',
     fp16=True,
)

#tracker = BYTETracker(

#)

# Selección del modelo

model = YOLO(detector_path)
num_frame = 0 # Identifica el frame para el print por consola. no tiene ninguna otra labor lógica
duracion_promedio = 0 # Para el cálculo de FPS promedio. Mejor explicado posteriormente
a = 0
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

    fin_time = time.time()
    a += 1
    # Traigo este cálculo aquí para poder aprovechar la variable a en el cálculo de la duración promedio.
    duracion = fin_time - ini_time  # Determinación del lapso transcurrido
    duracion_promedio = (duracion_promedio * a + duracion) / (a + 1)  # calculo de media con el resultado de la nueva iteración

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

    # show image with bboxes, ids, classes and confidences
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', im)
    output_video.write(im)
    print('Frame ' + str(num_frame))
    num_frame += 1

    # break on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
output_video.release()
cv2.destroyAllWindows()

FPS_promedio = 1 / duracion_promedio
print('FPS promedio del tracker + detector (FPS)= ' + str(FPS_promedio))
