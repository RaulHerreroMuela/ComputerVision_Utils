from ultralytics.models.yolo.model import YOLO
import cv2
import matplotlib.pyplot as plt

video_path = '.../Escritorio/videos_pesos_GT/Videos_orig/video_2.mp4'
video = cv2.VideoCapture(video_path)

model = YOLO('.../Escritorio/videos_pesos_GT/Weights/v5/yolov8_x_1280_v5.pt')


results = model.track(source = video_path,
                      stream = True,
                      agnostic_nms = True,
                      iou = 0.5, 
                      show = False,
                      save = False,
                      persist = True,
                      tracker = '.../Escritorio/Trackers originales/Repositorio Ultralytics/ultralytics/ultralytics/ultralytics/cfg/trackers/bytetrack.yaml')

a = 0
cajas = []
for r in results:
    if a == 0: # cambiar el valor de a por el del nº del frame a pintar - 1.
        for b in r.boxes:
            cajas.append({"xywhn" : b.xywhn, "id":b.id})
        #break
    a += 1

#print(cajas)

def dibujar_cajas(image, cajas):
    for caja in cajas:
        xywhn = caja["xywhn"].squeeze().tolist()
        box_id = int(caja["id"].item())
        
        # Extracting center coordinates and box dimensions
        cx, cy, w, h = xywhn
        xmin = int((cx - w / 2) * image.shape[1])
        ymin = int((cy - h / 2) * image.shape[0])
        xmax = int((cx + w / 2) * image.shape[1])
        ymax = int((cy + h / 2) * image.shape[0])

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f'ID: {box_id}', (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Read the nth frame of the video
video.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = video.read()

# Draw boxes on the first frame
frame_con_cajas = dibujar_cajas(frame.copy(), cajas)

# Display the image with boxes
plt.imshow(cv2.cvtColor(frame_con_cajas, cv2.COLOR_BGR2RGB))
plt.show()
 
