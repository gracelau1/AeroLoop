![Alt text](pictures/banner2.png)
# AeroLoop
A waste management system to **salvage on-flight recyclables** on commercial flights, optimised for **minimal energy use, hardware cost and weight added** to aircraft, designed for the purpose of seperating International Catering Waste (ICW) from non-ICW, so that recyclables **do not have to be incinerated or landfilled**.

# System Architecture
![Alt text](pictures/SystemArchitecture.png)


# Component 2 of AeroLoop: Airline Meal Computer Vision Project
| Benchmarks the Machine Learning model trained must meet | Measures Undertaken |
|--|--|
| Image capture and processing pipeline must be optimised for **fine-grain pixel detection**, as food bits on meal trays might be too small *(too few pixels)* for detection. | After model is trained, and after exporting and conversion to INT8, configure in Inference Script: - Use <ins>**Tiling**</ins> and <ins>**optimised resolution**</ins>: 640×640 + tiling *(split each image captured by Raspberry Pi Cam into 3×3 tiles, then run inference on each of the 9 images)* <br> -Set <ins>**confidence interval to a lower value**</ins> *(eg. 15%)* makes the model surface weaker, low-confidence objects <br>
| Response time **must be <2s** so as to not delay tray collection by cabin crews. | After model is trained, at exporting step: - <ins>**Export to ONNX**</ins>, then use the Jetson Nano's **<ins>own native TensorRT tooling (trtexec/onnc2trt)**</ins> to quantize the trained model to INT8 *(much better to do the conversion from ONNX to INT8 natively on the Jetson using its own TensorRT version, as YOLOv8's engine export uses TensorRT Python bindings, which might not generate an engine optimized for Jetson’s architecture (ARM, CUDA versions))* <br>| 
| Able to **distinguish between more than 2 food classes**, not just ICW and non-ICW, as different countries have different ICW regulations *(eg. fruits are ICW in US, while not in EU)*. This is so that we would  be able to output the correct LED light *(green/red)* according to the country of destination. | Choose <ins>**Image Segmentation**</ins> over just Classification or Multi-label Classification (See Below)

## Labels
1. Meat Bits
2. Vegetable Bits
3. Fruit Bits
4. Dessert Leftover
5. Meat
6. Vegetable
7. Fruit
8. Dessert

# Process of training ML Model
1: <ins>**Collect dataset**</ins> (using a phone/digital camera) <br>
2: <ins>**Label** </ins> images collected <br>

<p align="center">
  <img src="pictures/labelling2.gif" width="600" alt="Labelling GIF" /><br>
  <i>Image Segmentation on Roboflow</i>
</p>

3: Do <ins>**Data Augmentation**</ins> on current dataset to expose model to visual variability of tiny food bits (lighting, size, angle, occlusion) <br>
4: <ins>**Train Model**</ins> (On laptop, Colab Notebook) as yolov8n-seg.pt (Nano-friendly model) with: **augmentations**, **val=True** (tracking model performance mid-training by ensuring validation is run during training, **patience=20** to prevent wasted time and overfitting due to model memorising training images and doing poorly on new data, **device=0** for GPU selection on Colab Notebook instead of local computer CPU (for faster training duration) <br>
5: <ins>**Export model**</ins> on Colab Notebook as onnx, upload onnx to Jetson Nano (4GB) and on Jetson Nano (4GB), convert onnx to TensorRT with trtexec (Quantize only on representative data, TensorRT does this for INT8) <br>
6: <ins>**Configure Image Capture pipeline**</ins>. Add: **Tiling** step, Optimise **resolution** - keep img resolution 640 or below, **Set lower confidence interval** - increase detections surfaced, including low-confidence ones which is useful for small objects like crumbs) <br>
7: <ins>**Run inference**</ins> on Jetson NanoRun real-time inference with Raspberry Pi Camera Module v2, **assess accuracy**, **retrained** if accuracy <95%. <br>
<p align="center">
  <img src="pictures/Setup.jpg" width="250" alt="Setup Pic" style="display: inline-block;"/><br>
  <i>Wireless Nvidia Jetson Nano (4GB) Setup</i></br>
</p>


# Repository contains files for deployment on aircraft:
1. YOLOv8_Training_Colab.ipynb — **Colab notebook** used to train dataset
2. inference.py -  Script for Nvidia Jetson Nano (4GB) to **run inference** on images captured by Raspberry Pi camera
3. utils.py: **Helper script** for tiling images (3×3) to improve detection of tiny food bits
4. frontend_ipad.html and backend.py - Generate flight_details .json file containing destination to enable cabin crew to **dynamically change** Jetson Nano's output result **according to destination** and possibility of changes to ICW regulations.


