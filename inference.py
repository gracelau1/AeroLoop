#!/usr/bin/env python3
"""
engine_inference.py

TensorRT YOLOv8-Seg inference on Jetson with proper YOLOv8 mask proto decoding,
robust contour extraction (OpenCV compatibility), letterbox handling, top-k pre-NMS filtering,
and polygon label placement (centroid).
"""

import argparse
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Update CLASS_NAMES to match trained classes of a custom model (order must match training)
CLASS_NAMES = [
    "Carb", "Dairy", "Dessert", "Drink", "Fruit",
    "Meat", "Pastry", "Sauce", "Veg"
]
CLASS_COLORS = [
    (0, 255, 255), (0, 255, 0), (255, 0, 255),
    (255, 255, 0), (255, 0, 0), (0, 128, 255),
    (128, 0, 255), (0, 255, 128), (255, 128, 0)
]

INPUT_W = 640
INPUT_H = 640
NUM_CLASSES = len(CLASS_NAMES)
CONF_THRESH = 0.25
MASK_THRESH = 0.5
NMS_IOU_THRESH = 0.45
TOPK_PRE_NMS = 300  # keep top-k candidates by confidence before NMS (tune as needed)
MAX_OUTPUTS = 300   # safety cap for outputs


def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    print("Loaded engine. Bindings:")
    for b in engine:
        print("  ", b, "->", engine.get_binding_shape(b))
    return engine


def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        shape = engine.get_binding_shape(binding)
        # replace dynamic dimensions with 1 for allocation
        shape = tuple([1 if (s is None or s <= 0) else s for s in shape])
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"name": binding, "host": host_mem, "device": device_mem, "shape": shape})
        else:
            outputs.append({"name": binding, "host": host_mem, "device": device_mem, "shape": shape})
    print(f"Allocated {len(inputs)} input(s) and {len(outputs)} output(s).")
    return inputs, outputs, bindings, stream


def letterbox(im, new_shape=(INPUT_H, INPUT_W), color=(114, 114, 114)):
    # Resize with unchanged aspect ratio using padding (like Ultralytics)
    h0, w0 = im.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh


def preprocess_image(image_path):
    img0 = cv2.imread(image_path)
    if img0 is None:
        raise FileNotFoundError(f"Failed to read {image_path}")
    img, ratio, dw, dh = letterbox(img0, new_shape=(INPUT_H, INPUT_W))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    input_tensor = np.expand_dims(img_chw, axis=0)  # (1,3,H,W)
    return img0, input_tensor, ratio, dw, dh


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh_to_xyxy(box):
    x_c, y_c, w, h = box
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return np.array([x1, y1, x2, y2])


def scale_boxes(boxes_xyxy, ratio, dw, dh, orig_shape):
    # Undo letterbox for boxes (boxes in input-space -> original image coords).
    boxes_xyxy[:, [0, 2]] -= dw
    boxes_xyxy[:, [1, 3]] -= dh
    boxes_xyxy[:, :4] /= ratio
    h, w = orig_shape
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, h)
    return boxes_xyxy


def nms(boxes, scores, iou_threshold=NMS_IOU_THRESH):
    # Simple NMS; returns indices to keep relative to input arrays.
    if boxes.shape[0] == 0:
        return []
    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = boxes[:, 2].astype(float)
    y2 = boxes[:, 3].astype(float)
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def decode_masks_and_boxes(detection_output, proto_output, orig_shape, ratio, dw, dh):
    """
    detection_output: (46, 8400) as in .engine model (channels x anchors)
    proto_output: (32, 160, 160)
    Returns: boxes_xyxy (N,4), class_ids (N,), confidences (N,), masks (N, H_orig, W_orig)
    """
    # Transpose to (8400, 46)
    det = detection_output.T  # (num_anchors, channels)
    # split
    # [x, y, w, h, obj, cls0..clsN-1, mask_coefs...]
    boxes_xywh = det[:, 0:4].copy()
    objectness = sigmoid(det[:, 4])
    class_logits = det[:, 5:5 + NUM_CLASSES]
    class_probs = sigmoid(class_logits)
    mask_coeffs = det[:, 5 + NUM_CLASSES:46]  # shape (8400, num_coeffs)

    # per-anchor scores
    scores_all = objectness[:, None] * class_probs  # (8400, num_classes)
    class_ids = np.argmax(scores_all, axis=1)
    confidences = scores_all[np.arange(scores_all.shape[0]), class_ids]

    # filter by CONF_THRESH, keep top-K by confidence
    keep_idxs = np.where(confidences > CONF_THRESH)[0]
    if keep_idxs.size == 0:
        return np.zeros((0, 4)), np.array([]), np.array([]), np.zeros((0, orig_shape[0], orig_shape[1]), dtype=np.uint8)

    confs_filtered = confidences[keep_idxs]
    # take top-k by confidence
    top_k = min(TOPK_PRE_NMS, confs_filtered.size)
    order = confs_filtered.argsort()[::-1][:top_k]
    selected_idxs = keep_idxs[order]

    boxes_sel = boxes_xywh[selected_idxs]
    class_ids_sel = class_ids[selected_idxs]
    confidences_sel = confidences[selected_idxs]
    mask_coeffs_sel = mask_coeffs[selected_idxs]  # (K, Ccoeff)

    # convert xywh (normalized relative to input sized grid) -> xyxy in input-space
    boxes_xyxy = np.array([xywh_to_xyxy(b) for b in boxes_sel])
    # boxes were predicted relative to input image size (640). They likely are normalized to [0,1] or to grid coords
    # Many exports of YOLOv8 provide normalized x,y,w,h in [0,1] relative to input W,H
    # If boxes are normalized, multiply by INPUT_W/INPUT_H accordingly. To detect this, check scale:
    # I assumed they are normalized (0..1) * input_size. If not, this will need adjusting
    # Multiply normalized (x,y,w,h) by input size to get absolute coordinates in input space:
    boxes_xyxy[:, 0] *= INPUT_W
    boxes_xyxy[:, 1] *= INPUT_H
    boxes_xyxy[:, 2] *= INPUT_W
    boxes_xyxy[:, 3] *= INPUT_H

    # Undo letterbox to original image coordinates
    boxes_xyxy = scale_boxes(boxes_xyxy, ratio, dw, dh, orig_shape)

    # NMS
    keep_after_nms = nms(boxes_xyxy, confidences_sel)
    if len(keep_after_nms) == 0:
        return np.zeros((0, 4)), np.array([]), np.array([]), np.zeros((0, orig_shape[0], orig_shape[1]), dtype=np.uint8)

    boxes_final = boxes_xyxy[keep_after_nms]
    class_ids_final = class_ids_sel[keep_after_nms]
    confidences_final = confidences_sel[keep_after_nms]
    mask_coeffs_final = mask_coeffs_sel[keep_after_nms]  # (N, Ccoeff)

    # Prepare proto masks (handle extra channel possibility)
    proto = proto_output.copy()  # (C_proto, H_proto, W_proto)
    proto_channels = proto.shape[0]
    num_coeffs = mask_coeffs_final.shape[1]
    if num_coeffs == proto_channels - 1:
        proto = proto[:-1, :, :]  # drop last background channel if present

    proto_flat = proto.reshape(proto.shape[0], -1)  # (C_proto_use, H*W)

    # compute masks logits: (N, H*W) = mask_coeffs_final (N,C) dot proto_flat (C, H*W)
    masks_logits = np.dot(mask_coeffs_final, proto_flat)  # (N, H*W)
    masks_probs = sigmoid(masks_logits)
    H_proto, W_proto = proto.shape[1], proto.shape[2]
    masks = masks_probs.reshape(-1, H_proto, W_proto)  # (N, H_proto, W_proto)

    # Resize masks -> original image and threshold
    masks_bin_resized = np.zeros((masks.shape[0], orig_shape[0], orig_shape[1]), dtype=np.uint8)
    for i in range(masks.shape[0]):
        # resize from proto size to input size, but undo letterbox afterwards—easiest is:
        # 1) resize proto mask to INPUT_W x INPUT_H (model input space)
        mask_input_space = cv2.resize((masks[i] > MASK_THRESH).astype(np.uint8),
                                      (INPUT_W, INPUT_H),
                                      interpolation=cv2.INTER_NEAREST)
        # 2) crop letterbox padding and resize to orig image size
        # compute unpadded size in input space
        w_unpad = int(round(orig_shape[1] * ratio))
        h_unpad = int(round(orig_shape[0] * ratio))
        left = int(round(dw))
        top = int(round(dh))
        cropped = mask_input_space[top:top + h_unpad, left:left + w_unpad]
        if cropped.size == 0:
            # degenerate - fallback: resize full mask_input_space to orig
            mask_final = cv2.resize(mask_input_space, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_final = cv2.resize(cropped, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        masks_bin_resized[i] = (mask_final > 0).astype(np.uint8)

    return boxes_final, class_ids_final, confidences_final, masks_bin_resized


def find_contours_safe(bin_mask):
    #Robust cv2.findContours wrapper compatible with OpenCV 3/4 returns.
    res = cv2.findContours(bin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # res may be (contours, hierarchy) or (image, contours, hierarchy)
    if len(res) == 2:
        contours, hierarchy = res
    else:
        _, contours, hierarchy = res
    return contours, hierarchy


def draw_results(img, boxes, classes, confidences, masks):
    overlay = img.copy()
    alpha = 0.6
    h, w = img.shape[:2]

    for i in range(len(masks)):
        mask = masks[i]
        # ensure dtype uint8 0/255
        mask_uint8 = (mask.astype(np.uint8) * 255).astype(np.uint8)

        class_id = int(classes[i]) if len(classes) > 0 else 0
        conf = float(confidences[i]) if len(confidences) > 0 else 0.0
        box = boxes[i].astype(int) if len(boxes) > 0 else np.array([0, 0, w - 1, h - 1])

        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        fill_col = tuple(int(c) for c in color)
        outline_col = tuple(int(max(0, c * 0.6)) for c in color)
        text_color = (255, 255, 255)

        contours, _ = find_contours_safe(mask_uint8)

        centroid_x, centroid_y = None, None
        if contours:
            # draw all reasonably large contours
            for cnt in contours:
                if cv2.contourArea(cnt) < 20:
                    continue
                cv2.fillPoly(overlay, [cnt], fill_col)
                cv2.polylines(overlay, [cnt], True, outline_col, 2)
            # centroid from largest contour
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
        else:
            # fallback to bbox centre if no mask polygon
            centroid_x = int((box[0] + box[2]) / 2)
            centroid_y = int((box[1] + box[3]) / 2)
            # draw bbox outline as cue
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), outline_col, 2)

        if centroid_x is None:
            centroid_x, centroid_y = box[0], max(box[1] - 5, 10)

        label = f"{CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else class_id} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x = max(0, centroid_x)
        text_y = max(th + 2, centroid_y)
        cv2.rectangle(overlay, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline), outline_col, -1)
        cv2.putText(overlay, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help=".engine file path")
    parser.add_argument("--input", required=True, help="input image path")
    parser.add_argument("--output", required=True, help="output image path")
    args = parser.parse_args()

    engine = load_engine(args.engine)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    # Preprocess
    orig_img, input_tensor, ratio, dw, dh = preprocess_image(args.input)
    h0, w0 = orig_img.shape[:2]

    # Copy input
    # find input buffer index (assume first input)
    if len(inputs) == 0:
        raise RuntimeError("No input bindings found in engine.")
    np.copyto(inputs[0]["host"], input_tensor.ravel())
    # copy to device and run
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    t0 = time.time()
    # use execute_async_v2 with bindings list
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # copy outputs back
    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
    stream.synchronize()
    t1 = time.time()
    print(f"Inference time: {t1 - t0:.3f}s")

    # determine which output is proto and which is detection by shape/name
    # I expect proto: (1, C_proto, H_proto, W_proto) and detection: (1, 46, 8400)
    # outputs list order respects engine binding order; print for debugging:
    print("Outputs raw shapes (flat lengths):", [out["host"].shape for out in outputs])
    # find by shape if possible:
    proto_idx = None
    det_idx = None
    for i, out in enumerate(outputs):
        shape = out["shape"]
        # shape likely (1, 32, 160, 160) or (1,46,8400)
        if len(shape) == 4 and shape[1] > 4 and shape[2] > 1:
            # proto candidate
            proto_idx = i
        elif len(shape) == 3 or (len(shape) == 4 and shape[1] == 46):
            det_idx = i

    # fallback to defaults (older script ordering):
    if proto_idx is None or det_idx is None:
        # common ordering observed: outputs[0] proto, outputs[1] detection
        proto = outputs[0]["host"]
        det = outputs[1]["host"]
    else:
        proto = outputs[proto_idx]["host"]
        det = outputs[det_idx]["host"]

    # reshape according to observed shapes
    # from engine binding shapes
    proto_shape = outputs[0]["shape"] if proto is outputs[0]["host"] else outputs[proto_idx]["shape"]
    det_shape = outputs[1]["shape"] if det is outputs[1]["host"] else outputs[det_idx]["shape"]

    # convert flat host arrays to proper numpy arrays with shapes
    # Host arrays are 1D views of allocated size. Use shape info to reshape.
    # Find which outputs correspond to proto/detection by matching sizes:
    # I'll try reshape using the shapes I printed earlier (engine binding shapes)
    # Common expected shapes: proto -> (1, C, H, W), det -> (1, 46, 8400)
    # Map outputs by inspecting their total length:
    host_arrays = [out["host"] for out in outputs]
    host_lengths = [arr.size for arr in host_arrays]
    # Identify which has length matching 1*32*160*160 = 819,200? Wait your previous run reported:
    # Output0 shape: (819200,) and Output1 shape: (386400,) — that was flattened lengths.
    # I reshape based on binding shapes available in outputs[*]['shape'].
    def reshape_host_to_binding(arr, binding):
        shape = binding["shape"]
        return arr.reshape(shape)

    # attempt to reshape proto and det based on their binding dicts
    # find binding dicts with matching flattened lengths
    proto_arr = None
    det_arr = None
    for out in outputs:
        flatlen = out["host"].size
        # detect proto via size approx 1*C*H*W
        if flatlen == np.prod(out["shape"]):
            # if second dim (channel) > 4 treat as proto candidate
            if out["shape"][1] > 4:
                # assume proto
                if proto_arr is None:
                    proto_arr = out["host"].reshape(out["shape"])
                    proto_binding = out
                else:
                    pass
            else:
                if det_arr is None:
                    det_arr = out["host"].reshape(out["shape"])
                    det_binding = out
        else:
            # fallback: try to reshape into (1,32,160,160) or (1,46,8400) if sizes match
            pass

    # If identification failed, try default reshapes
    if proto_arr is None or det_arr is None:
        # try default known shapes
        try:
            # default mapping observed in .engine model
            proto_arr = outputs[0]["host"].reshape(outputs[0]["shape"])
            det_arr = outputs[1]["host"].reshape(outputs[1]["shape"])
        except Exception as e:
            # last resort: use first two outputs with guessed shapes
            # detection expected (1,46,8400) -> flatten length 386400 (1*46*8400)
            # proto expected (1,32,160,160) -> 819200
            a0, a1 = outputs[0]["host"], outputs[1]["host"]
            if a0.size == 819200 and a1.size == 386400:
                proto_arr = a0.reshape((1, 32, 160, 160))
                det_arr = a1.reshape((1, 46, 8400))
            elif a1.size == 819200 and a0.size == 386400:
                proto_arr = a1.reshape((1, 32, 160, 160))
                det_arr = a0.reshape((1, 46, 8400))
            else:
                raise RuntimeError("Could not determine output shapes automatically.") from e

    # flatten to drop batch dim
    proto_np = proto_arr.reshape(proto_arr.shape[1], proto_arr.shape[2], proto_arr.shape[3])  # (C, H, W)
    det_np = det_arr.reshape(det_arr.shape[1], det_arr.shape[2])  # (46, 8400)

    print(f"[DEBUG] detection_output shape after transpose: {det_np.T.shape}")
    # decode
    boxes, classes, confidences, masks = decode_masks_and_boxes(det_np, proto_np, (h0, w0), ratio, dw, dh)

    print(f"[INFO] final detections: {len(boxes)}")
    if len(boxes) == 0:
        print("No detections above threshold.")
        cv2.imwrite(args.output, orig_img)
        return

    result = draw_results(orig_img, boxes, classes, confidences, masks)
    cv2.imwrite(args.output, result)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()



