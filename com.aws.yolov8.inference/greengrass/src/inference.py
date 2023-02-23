import threading, time, cv2, ultralytics, torch, numpy as np, json
from datetime import datetime, timezone
import onnxruntime as ort
from .greengrass_mqtt_ipc import GreengrassMqtt
from .tensorrt_utils import TrtModel
from .utils import IOUtils
from .sort import Sort

MODEL_HEIGHT, MODEL_WIDTH = 640, 640
io_utuls = IOUtils(conf=0.3, iou=0.5, max_det=300, agnostic_nms=False, classes=None)
mot_tracker = Sort(max_age=30, min_hits=15, iou_threshold=0.50)

# Camera Class for starting/stopping a camera
class Camera:
    def __init__(self, config: dict) -> None:
        self.camera_id = config['camera_id']
        self.cam = None
        if self.camera_id.isnumeric():
            self.camera_id = int(self.camera_id)
        self.cam = cv2.VideoCapture(self.camera_id)
    def get_frame(self):
        if self.camera_status: return self.cam.read()[1]
        else: return None
    def stop_camera(self) -> None:
        self.cam.release()
    def camera_status(self):
        if not self.cam.isOpened():
            self.cam = cv2.VideoCapture(self.camera_id)
        return self.cam.isOpened()

# Inference Class for setting up the inference model, running inference and generating outputs
class Inference:
    def __init__(self, client: GreengrassMqtt, config: dict) -> None:
        self.camera = Camera(config = config)
        self.is_start, self.is_pause, self.is_stop = False, False, False
        self.client = client
        self.model_loc = config['model_loc']
        self.model_type = None
        self.model = None
        self.fps = 0.0
        self.fps_arr = []
        self.model_input_shape, self.model_output_shape = [], []
        if '.pt' in self.model_loc: # if PyTorch Model
            self.model_type = 'pytorch'
            self.model = ultralytics.YOLO(self.model_loc)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model_input_shape = (1, 3, MODEL_HEIGHT, MODEL_WIDTH)
            self.model_output_shape = (1, 84, 8400)
            print('[Inference] Success: Using YOLOv8 PyTorch model for inference...')
        elif '.onnx' in self.model_loc: # if ONNX Model
            self.model_type = 'onnx'
            self.model = ort.InferenceSession(self.model_loc)
            self.model_input_shape = self.model.get_inputs()[0].shape
            self.model_output_shape = self.model.get_outputs()[0].shape
            print('[Inference] Success: Using YOLOv8 ONNX model for inference...')
        elif '.trt' in self.model_loc: # if TensorRT Model
            self.model_type = 'tensorrt'
            self.model = TrtModel(engine_path=self.model_loc, model_height=MODEL_HEIGHT, model_width=MODEL_WIDTH)
            self.model_input_shape = self.model.input_shape
            self.model_output_shape = self.model.output_shape
            print('[Inference] Success: Using YOLOv8 TensorRT model for inference...')
        else:
            print('[Inference] Error: No valid model was provided')
        
        print(f'Model Input Shape  = {self.model_input_shape}')
        print(f'Model Output Shape = {self.model_output_shape}')
        
        self.inference_thread = threading.Thread(target = self.infer)
        self.inference_thread.start()
    
    def start(self):
        self.is_start = True
        self.is_pause = False
        self.is_stop = False

    def pause(self):
        self.is_start = False
        self.is_pause = True
        self.is_stop = False
    
    def stop(self):
        self.is_start = False
        self.is_pause = False
        self.is_stop = True
        self.camera.stop_camera()

    def infer(self): 
        while True:
            if self.is_stop: break
            if self.is_pause: continue 
            if self.is_start:
                image_in = self.camera.get_frame()
                if image_in is None: continue

                orig_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
                image = cv2.resize(orig_image, (MODEL_HEIGHT, MODEL_WIDTH))
                out_results = None

                infer_start_time = time.time()

                if self.model_type == 'pytorch':
                    with torch.no_grad():
                        out_results = self.model.predict(source = image)
                elif self.model_type == 'onnx':
                    image = io_utuls.preprocess(image, input_range=[0, 1])
                    image = image.transpose([2,0,1])
                    image = image[np.newaxis, ...]
                    out_results = self.model.run(None, {'images': image.astype(np.float32)})[0]
                    out_results = torch.from_numpy(np.array(out_results).reshape(self.model_output_shape)).cpu()
                    out_results = io_utuls.postprocess(out_results, self.model_input_shape, image_in.shape)
                elif self.model_type == 'tensorrt':
                    image = io_utuls.preprocess(image, input_range=[0, 1])
                    image = image.transpose([2,0,1])
                    image = image[np.newaxis, ...]
                    out_results = self.model(image.astype(np.float32))[0]
                    out_results = torch.from_numpy(np.array(out_results).reshape(self.model_output_shape)).cpu()
                    out_results = io_utuls.postprocess(out_results, self.model_input_shape, image_in.shape)
                
                infer_end_time = time.time()

                fps = 1./(infer_end_time - infer_start_time)
                self.fps_arr.append(fps)
                if len(self.fps_arr)>100: self.fps_arr = self.fps_arr[-100:]
                self.fps = round(np.mean(self.fps_arr),2)
                
                message = {}
                message['UTC Time'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
                message['Inferene Time (s)'] = infer_end_time - infer_start_time
                message['FPS'] = self.fps
                message['Model Format'] = self.model_type.upper()

                trackers = []

                if out_results is not None and self.model_type == 'pytorch':
                    for result in out_results:
                        if len(result)==0: continue
                        if result.boxes:
                            message['Model Type'] = 'Object Detection'
                            message['Inference Output'] = result.boxes.numpy().data.tolist()
                        elif result.masks:
                            message['Model Type'] = 'Segmentation'
                            message['Inference Output'] = result.masks.numpy().data.tolist()
                        elif result.preds:
                            message['Model Type'] = 'Classification'
                            message['Inference Output'] = result.preds.numpy().tolist()
                        if len(result)>0:
                            trackers = mot_tracker.update(result)
                        message['Tracking'] = trackers
                elif out_results is not None and self.model_type == 'onnx':
                    for result in out_results:
                        message['Model Type'] = 'Object Detection'
                        message['Inference Output'] = result
                        if len(result)>0:
                            trackers = mot_tracker.update(result)
                        message['Tracking'] = trackers
                elif out_results is not None and self.model_type == 'tensorrt':
                    for result in out_results:
                        message['Model Type'] = 'Object Detection'
                        message['Inference Output'] = result
                        if len(result)>0:
                            trackers = mot_tracker.update(result)
                        message['Tracking'] = trackers
                
                if len(message['Inference Output'])>1000:
                    message['Inference Output'] = "TBD"

                try:
                    self.client.publish_message(message)
                except Exception as e:
                    print(f"[Inference] MQTT Exception: {str(e)}")
                    if 'AWS_ERROR_EVENT_STREAM_MESSAGE_FIELD_SIZE_EXCEEDED' in str(e):
                        message['Inference Output'] = 'TOO LARGE'
                        self.client.publish_message(message)