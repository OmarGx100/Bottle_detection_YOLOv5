# import needed dependencies

#import yolo 
from ultralytics import YOLO
# import torch 
import torch

#importing os
import os 

# import cv2 for image handling
import cv2

#import time
import time 

#importing mqtt dependencies
import paho.mqtt.client as mqtt


# DEFUALT MODEL FORMAT 
# model_path = os.path.join('.', 'runs', 'detect', 'train6', 'weights', 'best.pt')
# model = YOLO(model_path)
# # MODEL WITH NCNN FORMAT
model_ncnn_path = os.path.join('.', 'runs', 'detect', 'train6', 'weights','best.torchscript')
model_ncnn = YOLO(model_ncnn_path)
# MODEL WITH ONNX FORMAT 
# model_ncnn_path = os.path.join('.', 'runs', 'detect', 'train6', 'weights','best.onnx')
# model_onxx = YOLO(model_ncnn_path)




# create a client to send data to the plc
def start_connection_with_mqtt(client_id, MqttBroker):
    client  = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,client_id)
    client.connect(MqttBroker)
    return client

def publish_message(client, message, topic):
    client.publish(topic, message)
    print(f"Published: {message} to topic {topic}")
    #time.sleep(0.1)


node_red_client_new = start_connection_with_mqtt("bottle", "broker.hivemq.com")
def send_plc_signal(client, class_id, topic):
    publish_message(client, class_id, topic)





def send_predictions(frame, predictions):

   # Get image dimensions
    width = frame.shape[1]
    # Draw bounding boxes and labels on the image
    for pred in predictions:

        # Extracxting needed data from model predictions
        class_id, confidence = pred.boxes.data.tolist()[0][5], pred.boxes.data.tolist()[0][4]
        x_center = pred.boxes.xywhn.tolist()[0][0]
        if confidence > 0.7 :
            if x_center * width >= (width / 2) - 10 and x_center * width <= (width / 2) + 10 :
                # send 1 to the node-red and wait for acknowladgement then send the class id
                send_plc_signal(node_red_client_new, 1, "bottle_detected_new")
                send_plc_signal(node_red_client_new, class_id + 2, "bottle_color_new") 


cap = cv2.VideoCapture(1)
start_time = 0
end_time = 0
while True :

    ret, frame = cap.read()

    if not ret :
        break
    
    # doing forward path for the model to predict our bounding boxes
    with torch.inference_mode():
        results = model_ncnn(frame)

    # visualizting predictions
    if len(results[0].boxes.data) != 0 :
        send_predictions(frame, results)
    else :
        send_plc_signal(node_red_client_new, 0, "bottle_detected_new")
        send_plc_signal(node_red_client_new, 0, "bottle_color_new")
    
    #calculating fps 
    end_time = time.time()
    fps = (1/(end_time - start_time))
    fps = f"FPS : {fps:.2f}"
    # Writing fps on webcam
    cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (233, 100, 120), 2)
    # opening the web cam with predictions
    cv2.imshow("predictions", frame)
    start_time = end_time
    # if you press s then break the programm
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# release all allocated memory 
cap.release()

# destroying the windows
cv2.destroyAllWindows() 