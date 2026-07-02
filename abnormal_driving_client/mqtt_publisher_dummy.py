from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import paho.mqtt.client as mqtt

from model_definition import FEATURE_COLS


TOPIC = "vehicle/gps_data"
CLIENT_ID = "gps_publisher_dummy"


def build_base_message() -> dict[str, object]:
    message: dict[str, object] = {
        "vid_md5": "demo_vehicle_001",
        "Lng": 108.9402 + random.uniform(-0.01, 0.01),
        "Lat": 34.3416 + random.uniform(-0.01, 0.01),
        "gps时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gps速度": random.uniform(35, 65),
        "加速度": random.uniform(-0.5, 0.5),
        "曲折度": random.uniform(1.0, 1.2),
    }
    for feature in FEATURE_COLS:
        message[feature] = random.uniform(0, 1)

    message["海拔"] = random.uniform(350, 450)
    message["vss速度"] = message["gps速度"]
    message["ACC状态"] = random.choice(["ACC开", "ACC关"])
    message["与正北方向夹角"] = random.uniform(0, 360)
    return message


def build_message(scenario: str) -> dict[str, object]:
    message = build_base_message()
    if scenario == "normal":
        message["gps速度"] = random.uniform(30, 60)
        message["加速度"] = random.uniform(-0.5, 0.5)
        message["曲折度"] = random.uniform(1.0, 1.2)
    elif scenario == "speeding":
        message["gps速度"] = random.uniform(90, 115)
        message["加速度"] = random.uniform(-0.5, 0.5)
        message["曲折度"] = random.uniform(1.0, 1.2)
    elif scenario == "hard_acceleration":
        message["gps速度"] = random.uniform(60, 85)
        message["vss速度"] = message["gps速度"]
        message["加速度"] = random.choice([random.uniform(1.5, 3.0), random.uniform(-3.0, -1.7)])
        message["曲折度"] = random.uniform(1.0, 1.2)
    elif scenario == "zigzag":
        message["与正北方向夹角"] = random.choice([15, 95, 175, 260])
        message["加速度"] = random.uniform(-0.5, 0.5)
        message["曲折度"] = random.uniform(1.6, 2.2)
    return message


def connect_mqtt(broker: str, port: int) -> mqtt.Client:
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
    except Exception:
        client = mqtt.Client(client_id=CLIENT_ID)
    client.connect(broker, port)
    return client


def publish_loop(client: mqtt.Client, topic: str, scenario: str, delay: float) -> None:
    while True:
        message = build_message(scenario)
        payload = json.dumps(message, ensure_ascii=False)
        result = client.publish(topic, payload)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"sent to {topic}: {payload}")
        else:
            print(f"failed to send message to {topic}")
        time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish demo GPS messages to MQTT.")
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default=TOPIC)
    parser.add_argument(
        "--scenario",
        choices=["normal", "speeding", "hard_acceleration", "zigzag"],
        default="normal",
    )
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    client = connect_mqtt(args.broker, args.port)
    client.loop_start()
    try:
        publish_loop(client, args.topic, args.scenario, args.delay)
    except KeyboardInterrupt:
        print("publisher stopped")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
