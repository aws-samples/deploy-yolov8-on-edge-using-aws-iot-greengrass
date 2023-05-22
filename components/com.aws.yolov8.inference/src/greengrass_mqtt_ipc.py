from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
from awsiot.greengrasscoreipc.model import (
    IoTCoreMessage,
    QOS,
)
import json
import queue

MQTT_TIMEOUT = 5

class GreengrassMqtt:
    def __init__(self, config: dict):
        self.ipc_client = GreengrassCoreIPCClientV2()
        self.event_topic = config['event_topic']
        self.output_topic = config['output_topic']
        self.subscription_op = self.ipc_client.subscribe_to_iot_core(topic_name=self.output_topic, qos=QOS.AT_LEAST_ONCE, on_stream_event=self.message_handler)
        self.queue = queue.Queue()

    def __del__(self):
        self.subscription_op.close()

    def message_handler(self, msg: IoTCoreMessage):
        message = msg.message.payload
        self.queue.put(json.loads(message.decode('utf8')))

    def publish_message(self, message: dict):
        self.ipc_client.publish_to_iot_core(topic_name=self.event_topic, qos=QOS.AT_LEAST_ONCE, payload=json.dumps(message))
