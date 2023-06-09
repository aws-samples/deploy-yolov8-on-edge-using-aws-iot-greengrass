# from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
# from awsiot.greengrasscoreipc.model import (
#     IoTCoreMessage,
#     QOS,
# )
# import json
# import queue

# MQTT_TIMEOUT = 5

# class GreengrassMqtt:
#     def __init__(self, config: dict):
#         self.ipc_client = GreengrassCoreIPCClientV2()
#         self.event_topic = config['event_topic']
#         self.output_topic = config['output_topic']
#         self.subscription_op = self.ipc_client.subscribe_to_iot_core(topic_name=self.output_topic, qos=QOS.AT_LEAST_ONCE, on_stream_event=self.message_handler)
#         self.queue = queue.Queue()

#     def __del__(self):
#         self.subscription_op.close()

#     def message_handler(self, msg: IoTCoreMessage):
#         message = msg.message.payload
#         self.queue.put(json.loads(message.decode('utf8')))

#     def publish_message(self, message: dict):
#         self.ipc_client.publish_to_iot_core(topic_name=self.event_topic, qos=QOS.AT_LEAST_ONCE, payload=json.dumps(message))

import awsiot.greengrasscoreipc
import awsiot.greengrasscoreipc.client as client
from awsiot.greengrasscoreipc.model import (
    IoTCoreMessage,
    QOS,
    SubscribeToIoTCoreRequest,
    PublishToIoTCoreRequest
)
import traceback ,json, queue

MQTT_TIMEOUT = 5

class StreamHandler(client.SubscribeToIoTCoreStreamHandler):
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
    def on_stream_event(self, event: IoTCoreMessage) -> None:
        try:
            message = str(event.message.payload, "utf-8")
            topic_name = event.message.topic_name
            print(f"[StreamHandler] Received message {message} on topic {topic_name}")
            self.queue.put(json.loads(message))
        except:
            traceback.print_exc()
    def on_stream_error(self, error: Exception) -> bool:
        # Handle error.
        return True  # Return True to close stream, False to keep stream open.
    def on_stream_closed(self) -> None:
        # Handle close.
        pass

class GreengrassMqtt:
    def __init__(self, config: dict):
        self.ipc_client = awsiot.greengrasscoreipc.connect()
        qos = QOS.AT_MOST_ONCE
        self.event_topic = config['event_topic']
        self.output_topic = config['output_topic']

        # For inference events like start/pause/end
        self.request_in = SubscribeToIoTCoreRequest()
        self.request_in.topic_name = self.event_topic
        self.request_in.qos = qos
        self.handler = StreamHandler()
        self.queue = self.handler.queue
        self.mqtt_timeout = MQTT_TIMEOUT
        operation = self.ipc_client.new_subscribe_to_iot_core(self.handler)
        future_response = operation.activate(self.request_in)
        future_response.result(self.mqtt_timeout)

        # For inference output response
        self.request_output = PublishToIoTCoreRequest()
        self.request_output.topic_name = self.output_topic

    def publish_message(self, message: dict):
        self.request_output.payload = bytes(json.dumps(message), "utf-8")
        qos = QOS.AT_LEAST_ONCE
        self.request_output.qos = qos
        operation = self.ipc_client.new_publish_to_iot_core()
        operation.activate(self.request_output)
        future_response = operation.get_response()
        future_response.result(self.mqtt_timeout)
