#Import necessary libraries
from flask import Flask, render_template, Response
import os
import cv2
import torch
from fasterRcnn.models.create_fasterrcnn_model import create_model
import cv2
from fasterRcnn.utils.transforms import infer_transforms, resize
from fasterRcnn.utils.annotations import (
    inference_annotations, convert_detections
)
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from PIL import ImageFont, ImageDraw, Image
from sort.sort import *
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import datetime
import time
from my_utils.utils import *
import traceback

# Tracking algorithm
mot_tracker = Sort()

# OCR Model
reader = easyocr.Reader(['bn'], gpu=True)

# License Plate Detection Model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("model_results/faster_rcnn/best_model.pth")
NUM_CLASSES = checkpoint['data']['NC']
CLASSES = checkpoint['data']['CLASSES']
build_model = create_model[checkpoint['model_name']]
model = build_model(num_classes=NUM_CLASSES, coco_model=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

###################################################################
camera = cv2.VideoCapture("test_videos/flip2.mp4")
###################################################################

number_plate = ""
final_number_plate = {}
prev_blur = 0
prev_track_id = 0
fontpath = "fonts/kalpurush.ttf"
font = ImageFont.truetype(fontpath, 50)
canvas_width = 500
canvas_height = 400
text_x = 0
text_y = 0
text_np = ""
confidence_threshold = 0.1

max_plate_size = 13000
min_plate_size = 4500

track_car = {}

#Vehicle model
model_V = torch.hub.load('ultralytics/yolov5', 'custom', 
                       path='model_results/yolo_v5/best.pt')

clases_dict = {0:"Car", 1: "Bus", 2: "Bike", 3:"CNG", 4: "Truck"}

# Timer to save json after every 5 second
s_time = time.time()

SAVE_DATA = 1


while True:
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8) 
    success, frame = camera.read()  # reading camera frame
    frame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)
    # Save a single frame which will help to draw on images with coordinates
    # cv2.imwrite("test.jpg", frame)
    # break
    
    if not success:
        break

    act_image = frame.copy()
    plate_image = frame.copy()

    # Convert the frame to grayscale for easy processing by the models
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecting cars
    detections_car = vehicle_detection(model_V, frame, confidence_threshold)
    
    # Detecting license plates
    image = infer_transforms(frame)
    frame = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(frame.to(DEVICE))
    
    outputs = [{k: v.to(DEVICE) for k, v in t.items()} for t in outputs]    
    
    if len(outputs[0]["boxes"]) != 0:
        bbox = outputs[0]["boxes"].tolist()[0]
        bbox = [int(i) for i in bbox]
        conf = round(outputs[0]["scores"].tolist()[0], 2)
        detections_ = []
        if conf > 0.5:
            # Get the bbox coordinates
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            detections_.append([x1, y1, x2, y2, conf])
            
            # Track the bounding box
            track_ids = mot_tracker.update(np.asarray(detections_))
            zo = 5
            
            try:
                track_id = track_ids[0][-1]
                plate_bbox = track_ids[0][:4]
                
                if track_id not in final_number_plate:
                    final_number_plate[track_id] = {"conf":0, "text": "", "name": ""}
                
                # Crop the plate from the image
                cropped_plate = plate_image[y1-zo:y2+zo, x1-zo:x2+zo]
                
                # Ge the area of the lisence plate
                sh = cropped_plate.shape
                plate_area = sh[0]*sh[1]
                
                # if (plate_area < min_plate_size) or (plate_area > max_plate_size):
                #     br = show_img(act_image, canvas)
                #     if br:
                #         break
                #     continue
                # Reset the license plate canvas when a new License plate arrives
                if prev_track_id != track_id:
                    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                    prev_blur = 0
                    prev_track_id = track_id
                    
                for det in detections_car:
                    
                    # Get the bbox of the vehicles
                    xmin, ymin, xmax, ymax, conf_car, cls = det.tolist()
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    
                    label = f'Class {clases_dict[int(cls)]}: {conf_car:.2f}'
                    
                    if check_if_inside(plate_bbox, [xmin, ymin, xmax, ymax]):
                        track_car[track_id] = [xmin, ymin, xmax, ymax, clases_dict[int(cls)]]
                        label = f'Class {clases_dict[int(cls)]}: {conf_car:.2f}  :  {track_id}'
                    
                    cv2.rectangle(act_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                    cv2.putText(act_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                
                
                # Draw the bounding box on the number plate
                cv2.rectangle(act_image, (x1, y1), (x2, y2), (0, 255, 34), 2)
                
                # Check if the blurriness of the license plate improves
                # TODO: Uncomment to take more clear images
                blur = is_blurry(cropped_plate)
                if blur >= prev_blur:
                    prev_blur = blur
                    
                    # Extract the text fom the lisence plate
                    result = reader.readtext(cropped_plate)
                    
                    total_conf = 0
                    temp_str = ""
                    
                    # Get the text and total confidence
                    # print(len(result))
                    for text in result:
                        conf_p = round(text[-1], 2)
                        t = text[-2]
                        if conf_p > 0.2:
                            total_conf+=conf_p
                            temp_str += t
                        print(f"Text: {t}     Conf: {conf_p}")
                    
                    # if len(result) == 2:
                    #     conf_p = round(result[0][-1], 2)
                    #     if conf_p > upper_part:
                    #         total_conf+=conf_p
                    #         upper_part = conf_p
                    #         temp_str[0] = result[0][-2]
                            
                    #     conf_p = round(result[1][-1], 2)
                    #     if conf_p > lower_part:
                    #         total_conf+=conf_p
                    #         lower_part = conf_p
                    #         temp_str[1] = result[1][-2]

                    
                    if total_conf >= final_number_plate[track_id]["conf"] and len(temp_str) > 5:
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        final_number_plate[track_id]["conf"] = total_conf
                        final_number_plate[track_id]["text"] = temp_str
                        
                        if track_id in track_car and final_number_plate[track_id]["name"] == "":
                            image_name = f"{current_time}_{track_car[track_id][-1]}.jpg"
                            final_number_plate[track_id]["name"] = image_name
                        
                        if SAVE_DATA:
                            cv2.imwrite(f"static/plates/{image_name}", cropped_plate)
                            cv2.imwrite(f"static/vehicle/{image_name}", act_image)

                    
                    
                    # Extract the license plate and put it in the license plate canvas
                    image_width, image_height = cropped_plate.shape[1], cropped_plate.shape[0]
                    paste_x = (canvas_width - image_width) // 2
                    paste_y = (canvas_height - image_height) // 2
                    canvas[paste_y:paste_y+image_height, paste_x:paste_x+image_width] = cropped_plate
                    
                    # Write banla text on the image and the license plate canvas
                    font_lp = ImageFont.truetype(fontpath, 20)
                    img_pil_lp = Image.fromarray(canvas)
                    draw_lp = ImageDraw.Draw(img_pil_lp)
                    text_width, text_height = draw_lp.textsize(final_number_plate[track_id]["text"], font_lp)
                    text_x = paste_x + (image_width - text_width) // 2
                    text_y = paste_y + image_height + 10
                    draw_lp.text((text_x, text_y), final_number_plate[track_id]["text"], font=font_lp, fill=(0, 255, 0))
                    canvas = np.array(img_pil_lp)
                    cv2.imshow("License Plate", canvas)
                    
                # Write the bangla text on main frame
                font = ImageFont.truetype(fontpath, 30)
                img_pil = Image.fromarray(act_image)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x1, y2+50),  final_number_plate[track_id]["text"], font = font, fill = (0, 255, 0, 0))
                draw_lp.text((text_x, text_y), final_number_plate[track_id]["text"], font=font_lp, fill=(0, 255, 0))
                act_image = np.array(img_pil)
            except Exception as e:
                print(e)
                        
            
            
                
            if SAVE_DATA:
                if time.time() - s_time > 5:
                    s_time = time.time()
                    with open("static/number_plate.json", "w") as outfile:
                        json.dump(final_number_plate, outfile)
                            
            
                # traceback.print_exc()
                
    br = show_img(act_image)
    if br:
        break
            