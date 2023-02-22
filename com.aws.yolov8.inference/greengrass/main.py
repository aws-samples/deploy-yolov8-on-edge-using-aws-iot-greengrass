import sys, json,  argparse, queue
from src.greengrass_mqtt_ipc import GreengrassMqtt
from src.inference import Inference

def main(config : dict):    
    client = GreengrassMqtt(config = config)
    inference = Inference(client = client, config = config)

    event_status = None

    # Keep the main thread alive, or the process will exit
    while True:
        message = client.queue.get()
        print(f"[Main] Received message {message}")
        if 'status' in message:
            if message['status'].lower() == 'start':
                event_status = 'STARTED'            
                print(f"[MAIN]: Inference {event_status}")
                inference.start()
            elif message['status'].lower() == 'pause':                
                event_status = 'PAUSED'            
                print(f"[MAIN]: Inference {event_status}")
                inference.pause()
            elif message['status'].lower() == 'stop':    
                event_status = 'STOPPED'            
                print(f"[MAIN]: Inference {event_status}")
                inference.stop()
                break

    # To stop subscribing, close the operation stream.
    operation.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    print("[Main] Arguments - ", sys.argv[1:])
    parser.add_argument("--config", type=str,required = True)
    args = parser.parse_args()
    print("[Main] ConfigRaw - ", args.config)        
    config = json.loads(args.config)
    print("[Main] ConfigJson - ", config) 
    main(config)