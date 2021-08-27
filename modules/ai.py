"""
Made by Project Lexus Team
Name: ai.py
Purpose: Talks with the AI and runs it.

Author: Tuna Cici
Created: 25/08/2021
"""

import os
import random
import glob
import cv2
import numpy
import time

from ctypes import *

if __name__ == "modules." + os.path.basename(__file__)[:-3]:
    # importing from outside the package
    from modules import config
    from modules import logger
else:
    # importing from main and inside the package
    import config
    import logger

# C Type Structures
class Box(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class Detection(Structure):
    _fields_ = [("bbox", Box),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embeddings_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DetNumPair(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(Detection))]

class Image(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class MetaData(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

def network_width(net):
    return lib.network_width(net)

def network_height(net):
    return lib.network_height(net)

def bbox2points(bbox):
    """
    Converts yolo type bounding box 
    to cv2 rectangle.
    """
    x, y, w, h = bbox
    x_min = int(round(x - (w / 2)))
    x_max = int(round(x + (w / 2)))
    y_min = int(round(y - (h / 2)))
    y_max = int(round(y + (h / 2)))

    return x_min, y_min, x_max, y_max

def class_colors(names):
    """
    Creates random color for each class name.
    Colorformat is BGR.
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}

def load_network(config_file: str, data_file : str,
                    weights : str, batch_size : int = 1):
    """
    loads model description and weights.
    takes:
        config_file: path to .cfg file.
        data_file: path to .data file
        weights: path to weights
    returns:
        network: the model
        class_names
        class_colors
    """
    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)
    metadata = load_meta(data_file.encode("ascii"))
    class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
    colors = class_colors(class_names)

    return network, class_names, colors

def print_detections(detections, coordinates = False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))

def draw_boxes(detections, image, colors):
    import cv2

    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cv2.rectangle(image,
                        (left, top),
                        (right, bottom),
                        colors[label],
                        1)
        cv2.putText(image,
                    "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    colors[label],
                    2)
    
    return image

def decode_detection(detections):
    decoded = []

    for label, confidence, bbox in detections:
        confidence = str(round(confidence * 100, 2))
        decoded.append((str(label), confidence, bbox))
    
    return decoded

def remove_negatives(detections, class_names, num):
    """
    removes all classes with 0 percent confidence
    """
    predictions = []

    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append(
                    (name, detections[j].prob[idx], (bbox))
                )

    return predictions

def detect_image(
    network,
    class_names,
    image,
    thresh = .5,
    hier_thresh = .5,
    nms = .45,):
    """
    runs the image throught the model and returns a list
    with highest confidence class and their bounding box
    """

    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(
        network,
        image.w,
        image.h,
        thresh,
        hier_thresh,
        None,
        0,
        pnum,
        0)
    num = pnum[0]
    if nms:
        do_nms_sort(
            detections,
            num,
            len(class_names),
            nms)
    predictions = remove_negatives(detections, class_names, num)
    predictions = decode_detection(predictions)
    free_detections(detections, num)

    return sorted(predictions, key=lambda x: x[1])

hasGPU = True

if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ";" + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()

    for k, v in os.environ.items():
        envKeys.append(k)

    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value {} not forcing CPU mode".format(tmp))
        except KeyError:
            # We never set the flag
            if "CUDA_VISIBLE_DEVICES" in envKeys:
                if int(os.environ["CUDA_VISIBLE_DEVICES"]) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError as cpu_error:
                print(cpu_error)
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except:
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(winNoGPUdll))
else:
    lib = CDLL("./libdarknet.so", RTLD_GLOBAL)

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [Image ,c_char_p]

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu


make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = Image

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(Detection)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(Detection)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(Detection), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DetNumPair), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(Detection), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(Detection), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [Image]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [Image, c_int, c_int]
letterbox_image.restype = Image

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = MetaData

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = Image

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [Image]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, Image]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, Image]
predict_image_letterbox.restype = POINTER(c_float)


network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, Image, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DetNumPair)

# ----------------------------------------------------------------------------

input_file = ""
batch_size = 1
weights = "yolov4.weights"
config_file = "cfg/yolov4.cfg"
data_file = "cfg/coco.data"
thresh = 0.25

def check_batch_shape(images, batch_size):
    """
    all the images must have the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images do not have the same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size is smaller than number of images")
    
    return shapes[0]

def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))

def prepare_batch(images, network, channels=3):
    width = network_width(network)
    height = network_height(network)

    darknet_images = []

    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                    interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)
    
    batch_array = numpy.concatenate(darknet_images, axis=0)
    batch_array = numpy.ascontiguousarray(batch_array.flat, dtype=numpy.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(POINTER(c_float))
    return Image(width, height, channels, darknet_images)

def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet does not accept numpy images
    # Create one with image we reuse for each detect
    width = network_width(network)
    height = network_height(network)
    darknet_image = make_image(width, height, 3)
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    
    copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = detect_image(network, class_names, darknet_image, thresh=thresh)
    free_image(darknet_image)
    image = draw_boxes(detections, image_resized, class_colors)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=0.5, nms=0.45, batch_size=4):
    image_height, image_width = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = network_predict_batch(network, darknet_images, batch_size, image_width,
                                                image_height, thresh, hier_thresh, None, 0, 0)
    
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            do_nms_obj(detections, num, len(class_names), nms)
        predictions = remove_negatives(detections, class_names, num)
        images[idx] = draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)

    free_batch_detections(batch_detections, batch_size)

    return images, batch_predictions

def image_classification(image, network, class_names):
    width = network_width(network)
    height = network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = make_image(width, height, 3)
    copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    free_image(darknet_image)

    return sorted(predictions, key=lambda x: -x[-1])

def convert2relative(image, bbox):
    """
    convert to relative coordinates for the annotion
    YOLO only uses this format
    """
    x, y, w, h = bbox
    height, width , _ = image.shape

    return x/width, y/height, w/width, h/height

def save_annotions(name, image, detections, class_names):
    """
    files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))

def main():
    random.seed(3)
    network, class_names, class_colors = load_network(
        config_file,
        data_file,
        weights,
        batch_size
    )

    images = load_images(input_file)

    index = 0
    while True:
        # loop asking for new image paths if no list is given
        if input_file:
            if index >= len(images):
                break
            image_name = images[index]
        else:
            image_name = input("Enter Image Path: ")

        prev_time = time.time()
        image, detections = image_detection(
            image_name, network, class_names, class_colors, thresh
        )
        
        print_detections(detections, False)
        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        cv2.imshow('Inference', image)
        if cv2.waitKey() & 0xFF == ord('q'):
            break     
        index += 1

main()
class Lexus_AI():
    bru = None
    

