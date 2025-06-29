import tensorrt as trt #using exported trained TensorRT model for optimized inference
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA driver

import cv2 # to capture video frames and manipulate images
import time # to measure inference time
import numpy as np
from gpiozero import LED # to control LEDs
import argparse
import os
 
from utils import tile_image #Imports tile_image() function from utils.py to split image into 3×3 grid for detecting small food bits

# destination variable change from crew updating flight info on iPad
import json
def get_flight_details(file_path="flight_details.json"):
    try:
        with open(file_path, "r") as f:
            details = json.load(f)
        return details
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using default flight details.")
        return {"destination": "US"}
    except json.JSONDecodeError:
        print(f"Warning: {file_path} corrupted or invalid JSON. Using default flight details.")
        return {"destination": "US"}

details = get_flight_details()
destination = details.get("destination", "US")



TRT_LOGGER = trt.Logger()
# Setup GPIO for LEDs (Jetson Nano pins)
red_led= LED(17)
green_led= LED(27)

# TensorRT engine loading and buffer allocation
def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
    return inputs, outputs, bindings, stream

def preprocess(image, input_shape):
    # Resize, normalize, transpose to CHW, expand batch dim
    img = cv2.resize(image, (input_shape[2], input_shape[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(output):
    # TODO: decode YOLOv8 output format to get detected classes
    # For now, assume output is raw logits or bbox/class/confidence data
    # Return a set of detected class names, e.g. {"meat", "fruit"}
    detected_classes = set()
    # Implement YOLOv8 output parsing here
    return detected_classes

def do_inference(context, bindings, inputs, outputs, stream):
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()
    return outputs[0]['host']

# Load TensorRT engine once
engine = load_engine("best_int8.trt")
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)
input_shape = engine.get_binding_shape(0)  # e.g. (3, 640, 640)



def decide_led(detected_classes, destination): #Use set of detected class names, decides which LED to turn on
    if destination== "US":
        return "red" if ("meat" in detected_classes or "fruit" in detected_classes) else "green"
    elif destination== "EU":
        return "red" if "meat" in detected_classes else "green"
    # else:
        # add Australia next



def run_tile_inference(destination, image_path, conf=0.15, overlap=30):
    tiles = tile_image(image_path, overlap=overlap)

    # Run batch inference on all tiles
    start = time.time()
    detected_classes = set()
    for tile in tiles:
        img_input = preprocess(tile, input_shape)
        np.copyto(inputs[0]['host'], img_input.ravel())
        output = do_inference(context, bindings, inputs, outputs, stream)
        detected_classes.update(postprocess(output))
    end = time.time()

    led_color = decide_led(detected_classes, destination)

    print(f"Inference time on {len(tiles)} tiles: {end - start:.2f} seconds")
    print(f"Detected classes: {detected_classes}")
    print(f"LED output: {led_color}")

    if led_color == "red":
        red_led.on()
        green_led.off()
    elif led_color == "green":
        green_led.on()
        red_led.off()
    else:
        red_led.off()
        green_led.off()
    return detected_classes, led_color



#captures frames from camera continuously and calls above function repeatedly
def run_live_video(destination, conf=0.15, overlap=30, cam_index=0):
    cap= cv2.VideoCapture(cam_index) # Open Pi Cam
    if not cap.isOpened(): #Exit if camera not found
        print("Error: Camera not accessible.")
        return
    
    #Continuously capture frames from camera
    print("Starting live inference. Press Ctrl+C to stop.")
    try:
        while True:
            destination = details.get("destination", "US") #delete if energy-intensive, but this is in case crew changes JSON file while inference script is running, pick up on new destination without restarting script
            ret, frame = cap.read()
            if not ret:
                print("No frame captured.")
                break

            # Save temp image so it can be tiled and processed like a static image
            temp_image_path = "live_frame.jpg"
            cv2.imwrite(temp_image_path, frame)

            # Call function for inference on tiled frame, over and over
            run_tile_inference(destination, temp_image_path, conf=conf, overlap=overlap)

            # Display (for troubleshooting)
            cv2.imshow("Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        red_led.off()
        green_led.off()





# set inference.py to auto-run on boot (via a systemd service,so that Jetson Nano will execute the script upon powering on ==> inference starts immediately after power-on
if __name__== "__main__":
    run_live_video(conf=0.15, overlap=30, country=destination, cam_index=0)


# for troubleshooting from terminal
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image", help="Path to image for single-frame inference")
    # parser.add_argument("--live", action="store_true", help="Use this flag to run live camera inference")
    # parser.add_argument("--conf", type=float, default=0.15)
    # parser.add_argument("--overlap", type=int, default=30)
    # parser.add_argument("--country", default="US")
    # parser.add_argument("--cam_index", type=int, default=0, help="Camera index (default 0)")

    # args = parser.parse_args()

    # if args.live:
    #     run_live_video(conf=args.conf, overlap=args.overlap, country=args.country, cam_index=args.cam_index)
    # elif args.image:
    #     run_tile_inference(args.image, conf=args.conf, overlap=args.overlap, country=args.country)
    # else:
    #     print("Please specify either --image or --live")


