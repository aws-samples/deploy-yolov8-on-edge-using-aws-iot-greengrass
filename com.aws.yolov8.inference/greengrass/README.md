# YOLOv8 Inference Greengrass component

---

## com.aws.yolov8.inference
- This component is used to run YOLOv8 on an edge device. The inference component runs the models stored in the `greengrass/models` directory. Whichever model and camera needs to be run has to be identified in the `greengrass/recipe.json` file. The following shows an example of the `camera_id` and `model_loc` inside the `greengrass/recipe.json` file:
```
    "Configuration": {
        "event_topic": "inference/input",
        "output_topic": "inference/output",
        "camera_id": "0",
        "model_loc": "models/yolov8n.trt"
    }
```
- Once the component is built, published and a deployment is run on the edge device, the ML model can be managed using MQTT Test Client.
- Subscribe to `inference/output` topic on the MQTT Test Client.
- Publish the status message on `inference/input` to either `start`, `pause` or `stop` inference processes.
- Once the inference is started, the inference results will start showing messages on the published topics.

The MQTT Messages Subscription/Publishing are as follows:
![MQTTMessages](../assets/MQTTMessages.png)