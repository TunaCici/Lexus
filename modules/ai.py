"""
Made by Project Lexus Team
Name: ai.py
Purpose: Initializez AI and DLL's.
Also, provides communication with it.

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
from typing import Callable

if __name__ == "modules." + os.path.basename(__file__)[:-3]:
    # importing from outside the package
    from modules import config
    from modules import logger
else:
    # importing from main and inside the package
    import config
    import logger

custom_logger = logger.LexusLogger()
# custom_logger.stop()

# C Type Structures
class Box(Structure):
    """
    C type box structure.
    x (float): left coordinate of the box.
    y (float): top coordinate of the box.
    w (float): width of the box.
    h (float): height of the box.
    """
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class Detection(Structure):
    """
    C type detection structure.
    bbox (box): bounding box.
    classes (int): number of the class
    prob (float*): probility
    """
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
    """
    C type detnumpair structure.
    num (int): class number of the object
    dets (detection): detects belonging to that object 
    """
    _fields_ = [("num", c_int),
                ("dets", POINTER(Detection))]

class Image(Structure):
    """
    C type image structure.
    w (int): width
    h (int): height
    c (channels): no. of color channels
    data (float*): image data
    """
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class MetaData(Structure):
    """
    C type metadata structure.
    classes (int): number of classes
    names (char*): class names
    """
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

# ----------------------------------------------------------------------------

class Lexus_AI():
    """
    wrapper class for the AI
    """
    
    input_file = ""
    batch_size = 1
    weights = "config/yolov4-tiny.weights"
    config_file = "config/yolov4-tiny.cfg"
    data_file = "config/coco.data"
    thresh = 0.25

    network: int
    class_names: list
    class_colors: list

    last_image: None
    last_detections: list

    hasGPU = True

    lib: CDLL
    copy_image_from_bytes: None
    predict: None
    set_gpu: None
    init_cpu: None
    make_image: None
    get_network_boxes: None
    make_network_boxes: None
    free_detections: None
    free_batch_detections: None
    free_ptrs: None
    network_predict: None
    reset_rnn: None
    load_net: None
    load_net_custom: None
    do_nms_obj: None
    do_nms_sort: None
    free_image: None
    letterbox_image: None
    load_meta: None
    load_image: None
    rgbgr_image: None
    predict_image: None
    predict_image_letterbox: None
    network_predict_batch: None

    def __init__(self):
        """
        loads the dll and the core C functions
        """
        
        custom_logger.log_info("Initialazing...")

        custom_logger.log_info("Importing DLL(s)...")
        # import the dlls
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
                self.lib = CDLL(winGPUdll, RTLD_GLOBAL)
            except:
                self.hasGPU = False
                if os.path.exists(winNoGPUdll):
                    self.lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
                    print("Notice: CPU-only mode")
                else:
                    # Try the other way, in case no_gpu was compile but not renamed
                    self.lib = CDLL(winGPUdll, RTLD_GLOBAL)
                    print("Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(winNoGPUdll))
        else:
            lib = CDLL("./libdarknet.so", RTLD_GLOBAL)

        # import the c functions from the dll
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.copy_image_from_bytes = self.lib.copy_image_from_bytes
        self.copy_image_from_bytes.argtypes = [Image ,c_char_p]

        self.predict = self.lib.network_predict_ptr
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        if self.hasGPU:
            self.set_gpu = self.lib.cuda_set_device
            self.set_gpu.argtypes = [c_int]

        self.init_cpu = self.lib.init_cpu

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = Image

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
        self.get_network_boxes.restype = POINTER(Detection)

        self.make_network_boxes = self.lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(Detection)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(Detection), c_int]

        self.free_batch_detections = self.lib.free_batch_detections
        self.free_batch_detections.argtypes = [POINTER(DetNumPair), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict_ptr
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.load_net_custom = self.lib.load_network_custom
        self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self.load_net_custom.restype = c_void_p

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(Detection), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(Detection), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [Image]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [Image, c_int, c_int]
        self.letterbox_image.restype = Image

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = MetaData

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = Image

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [Image]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, Image]
        self.predict_image.restype = POINTER(c_float)

        self.predict_image_letterbox = self.lib.network_predict_image_letterbox
        self.predict_image_letterbox.argtypes = [c_void_p, Image]
        self.predict_image_letterbox.restype = POINTER(c_float)


        self.network_predict_batch = self.lib.network_predict_batch
        self.network_predict_batch.argtypes = [c_void_p, Image, c_int, c_int, c_int,
                                        c_float, c_float, POINTER(c_int), c_int, c_int]
        self.network_predict_batch.restype = POINTER(DetNumPair)
        custom_logger.log_info("Import complete.")

        custom_logger.log_info("Loading the network...")
        random.seed(3)
        self.network, self.class_names, self.class_colors = self.load_network(
            self.config_file,
            self.data_file,
            self.weights,
            self.batch_size
        )
        custom_logger.log_info("Network loaded.")

        custom_logger.log_info("Initialazing complete.")

    def network_width(self, net: int):
        custom_logger.log_info(f"[network_width] net is: {type(net)}")
        """
        uses C function network_width().\n
        args:
            net: network
        returns:
            width of the image in the network
        """
        return self.lib.network_width(net)

    def network_height(self, net: int):
        custom_logger.log_info(f"[network_height] net is: {type(net)}")
        """
        uses C function network_height().\n
        args:
            net: network
        returns:
            height of the image in the network
        """
        return self.lib.network_height(net)

    def bbox2points(self, bbox: tuple):
        custom_logger.log_info(f"[bbox2points] bbox is: {type(bbox)}")
        """
        converts yolo type bounding box 
        to cv2 rectangle.\n
        args:
            bounding box
        returns:
            tuple of coordinates
        """
        x, y, w, h = bbox
        x_min = int(round(x - (w / 2)))
        x_max = int(round(x + (w / 2)))
        y_min = int(round(y - (h / 2)))
        y_max = int(round(y + (h / 2)))

        return x_min, y_min, x_max, y_max
    
    def class_colors(self, names: list):
        custom_logger.log_info(f"[class_colors] names is: {type(names)}")
        """
        creates random color for each class name.
        colorformat is BGR.\n
        args:
            list of names
        returns:
            list of names with colors (BGR)
        """
        return {name: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)) for name in names}

    def load_network(self, config_file: str, data_file : str,
                        weights : str, batch_size : int = 1):
        """
        loads model description and weights.\n
        args:
            config_file: path to .cfg file.
            data_file: path to .data file
            weights: path to weights
        returns:
            network: the model
            class_names: list if obj names
            class_colors: list of obj colors
        """
        network = self.load_net_custom(
            config_file.encode("ascii"),
            weights.encode("ascii"), 0, batch_size)
        metadata = self.load_meta(data_file.encode("ascii"))
        class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
        colors = self.class_colors(class_names)

        return network, class_names, colors

    def print_detections(self, detections: list, coordinates : bool = False):
        custom_logger.log_info(f"[print_detections] detections is: {type(detections)}")
        """
        prints the detections onto the terminal.\n
        args:
            detections: list of detections
            coordinates: whether to print coords or not
        returns:
            None
        """
        print("\nObjects:")
        for label, confidence, bbox in detections:
            x, y, w, h = bbox
            if coordinates:
                print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
            else:
                print("{}: {}%".format(label, confidence))
    
    def draw_boxes(self, detections: list, image: numpy.ndarray, colors: dict):
        custom_logger.log_info(
            f"[draw_boxes] detections is: {type(detections)}, image is: {type(image)}, colors is: {type(colors)}")
        """
        draws the detections onto image using cv2.\n
        args:
            detections: list of detections
            image: image to be drawn onto
            colors: list of obj colors
        returns:
            image: drawn version of the image
        """
        import cv2

        for label, confidence, bbox in detections:
            left, top, right, bottom = self.bbox2points(bbox)
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
    
    def decode_detection(self, detections: list):
        custom_logger.log_info(
            f"[decode_detection] detections is: {type(detections)}")
        """
        turns 0-1 value range to 0-100 for confidence level.\n
        args:
            detections: list of detections
        returns:
            decoded: list of decoded detections
        """
        decoded = []

        for label, confidence, bbox in detections:
            confidence = str(round(confidence * 100, 2))
            decoded.append((str(label), confidence, bbox))
        
        return decoded
    
    def remove_negatives(self, detections: Detection, class_names: list, num: int):
        custom_logger.log_info(
            f"[remove_negatives] detections is: {type(detections)}, class_names is: {type(class_names)}, num is: {type(num)}")
        """
        removes all classes with 0 percent confidence level.\n
        args:
            detections: list of detections
            class_name: list of class names
            num: I AM NOT SURE
        return:
            predictions: clean list of detections
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
        self,
        network: int,
        class_names: list,
        image: Image,
        thresh: float= .5,
        hier_thresh: float = .5,
        nms: float = .45):
        custom_logger.log_info(
            f"[detect_image] network is: {type(network)}, class_names is: {type(class_names)}, image is: {type(image)}")
        """
        runs the image throught the model and returns a list
        with highest confidence class and their bounding box\n
        args:
            network: the model
            class_names: list of class names
            image: image to be processed
            thresh: threshold value for the prediction
            hier_thresh: hierarchy threshold for Yolo9000
            nms: for trimming down multiple boxes
        returns:
            predictions: sorted list of predictions
        """

        pnum = pointer(c_int(0))
        self.predict_image(network, image)
        detections = self.get_network_boxes(
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
            self.do_nms_sort(
                detections,
                num,
                len(class_names),
                nms)
        predictions = self.remove_negatives(detections, class_names, num)
        predictions = self.decode_detection(predictions)
        self.free_detections(detections, num)

        return sorted(predictions, key=lambda x: x[1])
    
    def load_images(self, images_path: str):
        custom_logger.log_info(
            f"[load_images] images_path is: {type(images_path)}")
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
    
    def prepare_batch(self, images, network, channels=3):
        custom_logger.log_info(
            f"[prepare_batch] images is: {type(images)}, network is: {type(network)}, channels is: {type(channels)}")
        """
        prepares list of images to be proccessed.\n
        args:
            images: list of images
            network: the model
            channel: num of color channels
        return:
            image_struct: list of images ready to be processed
        """
        width = self.network_width(network)
        height = self.network_height(network)

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
    
    def image_detection(self, image: None, network: int, class_names: list, class_colors: dict, thresh: float):
        custom_logger.log_info(
            f"[image_detection] image is: {type(image)}, network is: {type(network)}, class_names is: {type(class_names)} class_colors is: {type(class_colors)}, thresh is: {type(thresh)}")
        """
        runs the image through the model to make predictions.\n
        args:
            image: image retrieved from cv2.imread
            network: the model
            class_names: list of class names
            class_colors: list of class colors
            thresh: threshold for predictions
        returns:
            image: cv2 type image to be displayed
            detections: list of detections
        """
        # Darknet does not accept numpy images
        # Create one with image we reuse for each detect
        width = self.network_width(network)
        height = self.network_height(network)
        darknet_image = self.make_image(width, height, 3)
        
        # image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                    interpolation=cv2.INTER_LINEAR)
        
        self.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = self.detect_image(network, class_names, darknet_image, thresh=thresh)
        self.free_image(darknet_image)
        image = self.draw_boxes(detections, image_resized, class_colors)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

    def batch_detection(self, network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=0.5, nms=0.45, batch_size=4):
        custom_logger.log_info(
            f"[batch_detection] network is: {type(network)}, images is: {type(images)}, class_names is: {type(class_names)} class_colors is: {type(class_colors)}")                
        """
        runs batch of images through the model to make predictions.\n
        args:
            network: the model
            images: list of images
            class_names: list of class names
            class_colors: list of class colors
            thresh: threshold for predictions
            hier_thresh hierarchy threshold for Yolo9000
            nms: for trimming down multiple boxes
        returns:
            images: list of images to be displayed
            batch_predictions: list of predictions for each image in the batch
        """
        image_height, image_width = self.check_batch_shape(images, batch_size)
        darknet_images = self.prepare_batch(images, network)
        batch_detections = self.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                    image_height, thresh, hier_thresh, None, 0, 0)
        
        batch_predictions = []
        for idx in range(batch_size):
            num = batch_detections[idx].num
            detections = batch_detections[idx].dets
            if nms:
                self.do_nms_obj(detections, num, len(class_names), nms)
            predictions = self.remove_negatives(detections, class_names, num)
            images[idx] = self.draw_boxes(predictions, images[idx], class_colors)
            batch_predictions.append(predictions)

        self.free_batch_detections(batch_detections, batch_size)

        return images, batch_predictions

    def image_classification(self, image, network, class_names):
        custom_logger.log_info(
            f"[image_classification] image is: {type(image)}, network is: {type(network)}, class_names is: {type(class_names)}")
        """
        runs the image through the model for classification.\n
        args:
            image: image to be used
            network: the model
            class_names: list of class names
        returns:
            sorted list of predictions
        """
        width = self.network_width(network)
        height = self.network_height(network)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                    interpolation=cv2.INTER_LINEAR)
        darknet_image = self.make_image(width, height, 3)
        self.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = self.predict_image(network, darknet_image)
        predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
        self.free_image(darknet_image)

        return sorted(predictions, key=lambda x: -x[-1])

    def convert2relative(self, image, bbox):
        custom_logger.log_info(
            f"[convert2relative] image is: {type(image)}, bbox is: {type(bbox)}")
        """
        converts to relative coordinates for the annotions
        YOLO only uses this format\n
        args:
            image: used for metadata for bbox
            bbox: bounding box of the image
        returns:
            relative coordinates
        """
        x, y, w, h = bbox
        height, width , _ = image.shape

        return x/width, y/height, w/width, h/height
    
    def save_annotions(self, name, image, detections, class_names):
        custom_logger.log_info(
            f"[save_annotions] name is: {type(name)}, image is: {type(image)}, detections is: {type(detections)} class_names is: {type(class_names)}")
        """
        saves the detection result to a text file.\n
        args:
            name: name of the file to be saved
            image: image
            detections: list of detections
            class_names: list of class names
        returns:
            None
        """
        file_name = os.path.splitext(name)[0] + ".txt"
        with open(file_name, "w") as f:
            for label, confidence, bbox in detections:
                x, y, w, h = self.convert2relative(image, bbox)
                label = class_names.index(label)
                f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))

    def run_and_display(self, image):
        """
        runs the image through the model and displays predictions.\n
        args:
            None
        returns:
            None
        """

        prev_time = time.time()
        image, detections = self.image_detection(
            image, self.network, self.class_names, self.class_colors, self.thresh
        )

        self.last_image = image
        self.last_detections = detections
        
        self.print_detections(detections, True)
        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        cv2.imshow('Inference', image)
        if cv2.waitKey() & 0xFF == ord('q'):
            return
    
    def update(self, image):
        """
        runs the image through the model and saves the result.
        args:
            image: image itself
        returns: Last Image
        """
        custom_logger.log_info("Updating AI.")
        try:
            self.last_image, self.last_detections = self.image_detection(
                image, self.network, self.class_names, self.class_colors, self.thresh
            )
            return self.last_image,self.last_detections
        except Exception as e:
            custom_logger.log_warning("Something went wrong while updating.")
            custom_logger.log_warning(e)
        custom_logger.log_info("Update complete.")

    def get_image(self):
        """
        returns the image from the last run
        args:
            None
        returns:
            image: proccessed image
        """
        return self.last_image

    def get_detections(self):
        """
        returns the last detection results.
        args:
            None
        returns:
            detections: list of detections from the last run
        """
        return self.last_detections