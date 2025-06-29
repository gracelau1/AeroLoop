from ultralytics import YOLO
import cv2 # to capture video frames and manipulate images
import time # to measure inference time
 
from utils import tile_image #Imports tile_image() function from utils.py to split image into 3×3 grid for detecting small food bits
from gpiozero import LED
import argparse
import os


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



model= YOLO("best.pt")  # Replace actual path of trained model



# Setup GPIO for LEDs (Jetson Nano pins)
red_led= LED(17)
green_led= LED(27)

def decide_led(detected_classes, destination): #Use set of detected class names, decides which LED to turn on
    if destination== "US":
        return "red" if ("meat" in detected_classes or "fruit" in detected_classes) else "green"
    elif destination== "EU":
        return "red" if "meat" in detected_classes else "green"
    ## else:
        ## add Australia next



def run_tile_inference(destination, image_path, conf=0.15, overlap=30): #tiles image, runs yolo inference on each tile, calc inference time
    tiles= tile_image(image_path, overlap=overlap)
    
    #Run batch inference on all 9 tiles, measures time taken
    start= time.time()
    results = model(tiles, conf=conf)
    end= time.time()

    detected_classes= set()
    for result in results:
        detected_classes.update(result.names[int(cls)] for cls in result.boxes.cls)

    led_color= decide_led(detected_classes, destination)

    print(f"Inference time on 9 tiles: {end - start:.2f} seconds") #Check inference speed
    print(f"Detected classes: {detected_classes}") # Output results
    print(f"LED output: {led_color}") # Output results

    # Control LED
    if led_color=="red":
        red_led.on()
        green_led.off()
    elif led_color=="green":
        green_led.on()
        red_led.off()
    else:
        red_led.off()
        green_led.off()
    return results, led_color



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


## for troubleshooting from terminal
    ## parser = argparse.ArgumentParser()
    ## parser.add_argument("--image", help="Path to image for single-frame inference")
    ## parser.add_argument("--live", action="store_true", help="Use this flag to run live camera inference")
    ## parser.add_argument("--conf", type=float, default=0.15)
    ## parser.add_argument("--overlap", type=int, default=30)
    ## parser.add_argument("--country", default="US")
    ## parser.add_argument("--cam_index", type=int, default=0, help="Camera index (default 0)")

        ## args = parser.parse_args()

        ## if args.live:
            ## run_live_video(conf=args.conf, overlap=args.overlap, country=args.country, cam_index=args.cam_index)
        ## elif args.image:
            ## run_tile_inference(args.image, conf=args.conf, overlap=args.overlap, country=args.country)
        ## else:
            ## print("⚠️ Please specify either --image or --live")

