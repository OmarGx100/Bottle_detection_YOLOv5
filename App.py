from ultralytics import YOLO
import supervision as sv
import cv2
import paho.mqtt.client as mqtt
import numpy as np

# Path to the tranied model
model_path = './runs/detect/train6/weights/best.pt'

# Instantiating Model object
model = YOLO(model_path)


# a function to start connection with the MqttBroker given cliend id
def start_connection_with_mqtt(client_id, MqttBroker):
    client  = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,client_id)
    client.connect(MqttBroker, keepalive=40)
    return client


# Function to publish a message to a given topic using a specific client
def publish_message(client, message, topic):
    client.publish(topic, message, qos = 0)


# Function to stop connection between the client and the broker
def stop_connection(client):
    client.disconnect()
    print("connection had stopped")


# Instantiating a client
node_red_client = start_connection_with_mqtt("bottle_color_new", "broker.hivemq.com")
node_red_client.reconnect_delay_set(min_delay=1, max_delay=120)

# Reading frames from the camer and extract the frame information
cap = cv2.VideoCapture(0)
_, img = cap.read()
height, width, _ = img.shape


# Instantiating box_annotator object so we can draw bounding boxes on the detected objects
box_annotator = sv.BoundingBoxAnnotator()

# Instantiating Label annotatoe Object so we can labelize the detected object
label_annotator = sv.LabelAnnotator()


# Instantiating A tracker object to give each detected object an ID so we can later count them
tracker = sv.ByteTrack()
smoother = sv.DetectionsSmoother()


# 2 sv.Points to define the triggering line
LINE_START = sv.Point(3 * width/4, 0)
LINE_END  = sv.Point(3 * width/4, height)


# Counters Positions
RED_ANCHOR = sv.Point(width - 100, 30)
BLUE_ANCHOR = sv.Point(width - 100, 70)
YELLOW_ANCHOR = sv.Point(width - 100, 110)

# Starting counters
BLUE_COUNT = 0
RED_COUNT = 0
YELLOW_COUNT = 0
bbox_width = 0


# Instantiating blue, red, yellow lines for triggering counters
blue_line = sv.LineZone(LINE_START, LINE_END)
red_line = sv.LineZone(LINE_START, LINE_END)
yellow_line = sv.LineZone(LINE_START, LINE_END)
flag = 0

while True:
    # Reading camera frames
    ret, frame = cap.read()
    height, width, _ = frame.shape
    
    # While there is frames continue
    if not ret:
        break

    # Sending The frames to the model and reciving the model detections
    results = model(frame)[0]
    dict_names = results.names

    # Converting the detections into super vision detection object so we can deal with it easly
    detections = sv.Detections(xyxy = results.boxes.xyxy.cpu().numpy().astype(int),
                               class_id = results.boxes.cls.cpu().numpy().astype(int),
                               confidence = results.boxes.conf.cpu().numpy())
    
    # filtering our detection so we just take those that their confidence intervel greater than 70%
    detections = detections[detections.confidence > 0.6]


    # Updating the tracker with that detection so each detection can have a tracker id
    detections = tracker.update_with_detections(detections = detections)
    detections = smoother.update_with_detections(detections = detections)
    

    # Drawing a line for debugging purposes
    cv2.line(img = frame, pt1= (width//2, 0), pt2 = (width//2, height), color = (0, 0, 0), thickness = 3)
    
    
    # Looping across all the detections and check wether of them is in the triggering area or not 
    for detection in detections:
        bbox_width = detection[0][0] + int((detection[0][2] - detection[0][0])//2)

        # giving +or-3 pixels to have some safe zone 
        if (bbox_width <= width/2 + 5) & (bbox_width >= width/2 - 5):
            # we publish a message to the node-red if there is a bottle in the middle of the frame
            publish_message(node_red_client, int(detection[3]) + 2, "bottle_color_new")
            publish_message(node_red_client,1,"bottle_detected_new")    
        # else:
        #     if flag == 0:
        #         # we send 0 that means that there is not bottles in the middle of the frame
        #         publish_message(node_red_client, 0, "bottle_color_new")
        #         publish_message(node_red_client, 0, "bottle_detected_new")
        #         flag = 2
        #     else :
        #         publish_message(node_red_client, 2, "bottle_colo_new")
        #         # publish_message(node_red_client, ) 
        #         flag  = 0
    publish_message(node_red_client, 0, "bottle_color_new")
    publish_message(node_red_client, 0, "bottle_detected_new")

    # Filtering all detections into red, blue, yellow bottles so we can trigger the counters 
    red_detections = detections[detections.class_id == 1]
    blue_detections = detections[detections.class_id == 0]
    yellow_detections = detections[detections.class_id == 2]

    # Triggering the lines of the counters 
    blue_in, blue_out = blue_line.trigger(blue_detections)
    red_in, red_out = red_line.trigger(red_detections)
    yellow_in, yellow_out = yellow_line.trigger(yellow_detections)

    # checking wether a specific bottle crossed the line it belongs to or not if yes add one to the counter
    if np.any(blue_out):
        BLUE_COUNT += np.sum(blue_out)
    elif np.any(red_out):
        RED_COUNT += np.sum(red_out)
    elif np.any(yellow_out):
        YELLOW_COUNT += np.sum(yellow_out)
    
    
    # labels to be visualized for the counters
    red_counter_label  = f"RED COUNT : {RED_COUNT}"
    blue_counter_label = f"BLUE COUNT : {BLUE_COUNT}"
    yellow_counter_label = f"YELLOW COUNT : {YELLOW_COUNT}"
    

    # labels to be visualized on each detected bottle
    labels = [f"{dict_names[class_id]} {conf:.2f}"
              
              for class_id, conf in zip(detections.class_id, detections.confidence)]
    
    # Visualizing the counters on the frame
    frame = sv.draw_text(scene = frame, text = blue_counter_label, text_anchor= BLUE_ANCHOR,text_color= sv.Color.BLACK , background_color= sv.Color.WHITE)
    frame = sv.draw_text(scene = frame, text = red_counter_label, text_anchor= RED_ANCHOR,text_color= sv.Color.BLACK , background_color= sv.Color.WHITE)
    frame = sv.draw_text(scene = frame, text = yellow_counter_label, text_anchor= YELLOW_ANCHOR,text_color= sv.Color.BLACK , background_color= sv.Color.WHITE)
    frame = box_annotator.annotate(scene = frame, detections=detections)
    frame = label_annotator.annotate(scene = frame, detections=detections, labels=labels)

    # Showing the camera frames on the screen 
    cv2.imshow('predictions', frame)

    # if you pressed 's' on the keyboard the stream will stops
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# releasing the camera and destroying all windows after the stream stops to clear the storage
cap.release()
cv2.destroyAllWindows()


stop_connection(node_red_client)