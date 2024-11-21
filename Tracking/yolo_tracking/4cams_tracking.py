import cv2
import numpy as np
import time
from pathlib import Path

from boxmot import DeepOCSORT
from boxmot import BoTSORT
from boxmot import BYTETracker

from ultralytics.models.yolo.model import YOLO
import os

# Video path
video_cam1_path = '.../Escritorio/videos_pesos_GT/Videos_orig/4cam_sinc/output1_compress.mp4'
video_cam2_path = '.../Escritorio/videos_pesos_GT/Videos_orig/4cam_sinc/output2_compress.mp4'
video_cam3_path = '.../Escritorio/videos_pesos_GT/Videos_orig/4cam_sinc/output3_compress.mp4'
video_cam4_path = '.../Escritorio/videos_pesos_GT/Videos_orig/4cam_sinc/output4_compress.mp4'
video_4cam_stitch_path = '.../scritorio/videos_pesos_GT/Videos_orig/4cam_sinc/output_stitch.mp4'


# Grabación de video
output_video_path = '.../Escritorio/Resultados/videos_4cam/bytetracker_bruto.mp4'
fourcc = cv2.VideoWriter.fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (1920, 3452))

# Detector path
detector_path = '.../Escritorio/videos_pesos_GT/Weights/1cam/yolov8_m_640_v5.pt'

# Dimensiones imagenes DeepFarm
real_width = 1920
real_height_1cam = 1080
real_height_4cam = 3452

vid_1 = cv2.VideoCapture(video_cam1_path)
vid_2 = cv2.VideoCapture(video_cam2_path)
vid_3 = cv2.VideoCapture(video_cam3_path)
vid_4 = cv2.VideoCapture(video_cam4_path)
vid_stitch = cv2.VideoCapture(video_4cam_stitch_path)

color = (0, 0, 255)  # BGR
thickness = 2
fontscale: float = 2

#tracker = BoTSORT(
     #model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
#    model_weights=Path('.../Escritorio/ultralytics_github/Tracking_deep_farm/log/osnet_x1_0_050124/model/osnet_x1_0_DeepFarm050124.pt'),
#    device='cuda:0',
#     fp16=True,
#)

tracker = BYTETracker(

)

model = YOLO(detector_path)
num_frame = 0 # Identifica el frame para el print por consola. no tiene ninguna otra labor lógica
duracion_promedio = 0 # Para el cálculo de FPS promedio. Mejor explicado posteriormente
a = 0

while True:

    ini_time = time.time() # Se registra el instante del inicio de la iteración

    ret_cam1, im_cam1 = vid_1.read()
    ret_cam2, im_cam2 = vid_2.read()
    ret_cam3, im_cam3 = vid_3.read()
    ret_cam4, im_cam4 = vid_4.read()
    ret_camstitch, im_camstitch = vid_stitch.read()

    if (not ret_cam1 or not ret_cam2 or not ret_cam3 or not ret_cam4 or not ret_camstitch):
        print('-------------------------------------')
        print('----------- Fin del video -----------')
        print('-------------------------------------')
        print('')
        break

    # substitute by your object detector, input to tracker has to be N X (x, y, x, y, conf, cls)
    # Aquí vamos a hacer las detecciones sobre las 4 imagenes y luego las vamos a propagar al stitching

    l_results_1 = model(im_cam1)
    l_results_2 = model(im_cam2)
    l_results_3 = model(im_cam3)
    l_results_4 = model(im_cam4)

    num_bboxes_1 = l_results_1[0].boxes.xyxy.size(0)
    num_bboxes_2 = l_results_2[0].boxes.xyxy.size(0)
    num_bboxes_3 = l_results_3[0].boxes.xyxy.size(0)
    num_bboxes_4 = l_results_4[0].boxes.xyxy.size(0)
    num_bboxes = num_bboxes_1 + num_bboxes_2 + num_bboxes_3 + num_bboxes_4

    results = np.zeros((num_bboxes, 6), dtype='float32')

    # Logística de cuda, tensores, cpu, etc
    boxes_gpu_1 = l_results_1[0].boxes.xyxy
    boxes_cpu_1 = boxes_gpu_1.cpu()
    res_boxes_1 = boxes_cpu_1.numpy()

    boxes_gpu_2 = l_results_2[0].boxes.xyxy
    boxes_cpu_2 = boxes_gpu_2.cpu()
    res_boxes_2 = boxes_cpu_2.numpy()

    boxes_gpu_3 = l_results_3[0].boxes.xyxy
    boxes_cpu_3 = boxes_gpu_3.cpu()
    res_boxes_3 = boxes_cpu_3.numpy()

    boxes_gpu_4 = l_results_4[0].boxes.xyxy
    boxes_cpu_4 = boxes_gpu_4.cpu()
    res_boxes_4 = boxes_cpu_4.numpy()


    cls_gpu_1 = l_results_1[0].boxes.cls
    cls_cpu_1 = cls_gpu_1.cpu()
    res_cls_1 = cls_cpu_1.numpy()
    cls_gpu_2 = l_results_2[0].boxes.cls
    cls_cpu_2 = cls_gpu_2.cpu()
    res_cls_2 = cls_cpu_2.numpy()
    cls_gpu_3 = l_results_3[0].boxes.cls
    cls_cpu_3 = cls_gpu_3.cpu()
    res_cls_3 = cls_cpu_3.numpy()
    cls_gpu_4 = l_results_4[0].boxes.cls
    cls_cpu_4 = cls_gpu_4.cpu()
    res_cls_4 = cls_cpu_4.numpy()

    conf_gpu_1 = l_results_1[0].boxes.conf
    conf_gpu_1 = conf_gpu_1.cpu()
    res_conf_1 = conf_gpu_1.numpy()
    conf_gpu_2 = l_results_2[0].boxes.conf
    conf_gpu_2 = conf_gpu_2.cpu()
    res_conf_2 = conf_gpu_2.numpy()
    conf_gpu_3 = l_results_3[0].boxes.conf
    conf_gpu_3 = conf_gpu_3.cpu()
    res_conf_3 = conf_gpu_3.numpy()
    conf_gpu_4 = l_results_4[0].boxes.conf
    conf_gpu_4 = conf_gpu_4.cpu()
    res_conf_4 = conf_gpu_4.numpy()

    # Generación de results

    b = 0
    for t in range(num_bboxes_1): # Camara 1
        sup_izq = res_boxes_1[t, 0:2]
        inf_der = res_boxes_1[t, 2:4]

        sup_izq_trans = cv2.perspectiveTransform(sup_izq.reshape(1,1,2), H1)
        inf_der_trans = cv2.perspectiveTransform(inf_der.reshape(1,1,2), H1)

        results[b, 0:2] = sup_izq_trans
        results[b, 2:4] = inf_der_trans
        results[b, 4] = res_conf_1[t]
        results[b, 5] = res_cls_1[t]

        b += 1

    for t in range(num_bboxes_2): # Camara 2
        sup_izq = res_boxes_2[t, 0:2]
        inf_der = res_boxes_2[t, 2:4]

        sup_izq_desdist = cv2.perspectiveTransform(sup_izq.reshape(1, 1, 2), H2)
        sup_izq_trans = cv2.perspectiveTransform(sup_izq_desdist.reshape(1,1,2), H12)
        inf_der_desdist = cv2.perspectiveTransform(inf_der.reshape(1, 1, 2), H2)
        inf_der_trans = cv2.perspectiveTransform(inf_der_desdist.reshape(1, 1, 2), H12)

        results[b, 0:2] = sup_izq_trans
        results[b, 2:4] = inf_der_trans
        results[b, 4] = res_conf_2[t]
        results[b, 5] = res_cls_2[t]

        b += 1

    for t in range(num_bboxes_3):
        sup_izq = res_boxes_3[t, 0:2]
        inf_der = res_boxes_3[t, 2:4]

        sup_izq_desdist = cv2.perspectiveTransform(sup_izq.reshape(1, 1, 2), H3)
        sup_izq_trans = cv2.perspectiveTransform(sup_izq_desdist.reshape(1,1,2), H123)
        inf_der_desdist = cv2.perspectiveTransform(inf_der.reshape(1, 1, 2), H3)
        inf_der_trans = cv2.perspectiveTransform(inf_der_desdist.reshape(1, 1, 2), H123)

        results[b, 0:2] = sup_izq_trans
        results[b, 2:4] = inf_der_trans
        results[b, 4] = res_conf_3[t]
        results[b, 5] = res_cls_3[t]

        b += 1

    for t in range(num_bboxes_4):
        sup_izq = res_boxes_4[t, 0:2]
        inf_der = res_boxes_4[t, 2:4]

        sup_izq_desdist = cv2.perspectiveTransform(sup_izq.reshape(1, 1, 2), H4)
        sup_izq_trans = cv2.perspectiveTransform(sup_izq_desdist.reshape(1, 1, 2), H1234)
        inf_der_desdist = cv2.perspectiveTransform(inf_der.reshape(1, 1, 2), H4)
        inf_der_trans = cv2.perspectiveTransform(inf_der_desdist.reshape(1, 1, 2), H1234)

        results[b, 0:2] = sup_izq_trans
        results[b, 2:4] = inf_der_trans
        results[b, 4] = res_conf_4[t]
        results[b, 5] = res_cls_4[t]
        b += 1

    # tracking de results
    tracks = tracker.update(results, im_camstitch) # --> (x, y, x, y, id, conf, cls, ind)

    xyxys = tracks[:, 0:4].astype('int') # float64 to int
    ids = tracks[:, 4].astype('int') # float64 to int
    confs = tracks[:, 5]
    clss = tracks[:, 6].astype('int') # float64 to int
    inds = tracks[:, 7].astype('int') # float64 to int

    fin_time = time.time()
    a += 1

    duracion = fin_time - ini_time  # Determinación del lapso transcurrido
    duracion_promedio = (duracion_promedio * a + duracion) / (a + 1)  # calculo de media con el resultado de la nueva iteración

    # print bboxes with their associated id, cls and conf
    if tracks.shape[0] != 0:
        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
            im_camstitch = cv2.rectangle(
                im_camstitch,
                (xyxy[0], xyxy[1]),
                (xyxy[2], xyxy[3]),
                color,
                thickness
            )
            cv2.putText(
                im_camstitch,
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
    cv2.imshow('frame', im_camstitch)
    output_video.write(im_camstitch)
    print('Frame ' + str(num_frame))
    num_frame += 1

    # break on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid_1.release()
vid_2.release()
vid_3.release()
vid_4.release()
vid_stitch.release()
output_video.release()
cv2.destroyAllWindows()

FPS_promedio = 1 / duracion_promedio
print('FPS promedio del tracker + detector (FPS)= ' + str(FPS_promedio))