import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import rosbag
import numpy as np
import cv2
import json
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

KST = timezone(timedelta(hours=9))

def convert_timestamp(nanoseconds):
    seconds = nanoseconds / 1e9
    dt = datetime.fromtimestamp(seconds, tz=KST)
    formatted_time = dt.strftime('%Y-%m-%d_%H-%M-%S.%f')[:-4]
    return formatted_time

def odom_to_dict(msg: Odometry) -> dict:
    return {
        "header": {
            "seq": msg.header.seq,
            "stamp": {"secs": msg.header.stamp.secs, "nsecs": msg.header.stamp.nsecs},
            "frame_id": msg.header.frame_id,
        },
        "child_frame_id": msg.child_frame_id,
        "pose": {
            "position": {
                "x": msg.pose.pose.position.x,
                "y": msg.pose.pose.position.y,
                "z": msg.pose.pose.position.z,
            },
            "orientation": {
                "x": msg.pose.pose.orientation.x,
                "y": msg.pose.pose.orientation.y,
                "z": msg.pose.pose.orientation.z,
                "w": msg.pose.pose.orientation.w,
            },
            "covariance": list(msg.pose.covariance),
        },
        "twist": {
            "linear": {
                "x": msg.twist.twist.linear.x,
                "y": msg.twist.twist.linear.y,
                "z": msg.twist.twist.linear.z,
            },
            "angular": {
                "x": msg.twist.twist.angular.x,
                "y": msg.twist.twist.angular.y,
                "z": msg.twist.twist.angular.z,
            },
            "covariance": list(msg.twist.covariance),
        },
    }

def save_camera(msg, save_path):
    img = np.frombuffer(msg.data, np.uint8)
    img_bayer = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img_bayer, cv2.COLOR_BAYER_RG2RGB)
    cv2.imwrite(str(save_path), img_rgb)

def save_odom(msg, save_path):
    with open(save_path, "w") as f:
        json.dump(odom_to_dict(msg), f, indent=4)

def save_pointcloud2(msg, save_path):
    # Convert PointCloud2 to numpy array (x, y, z, intensity)
    points = []
    for p in pc2.read_points(msg, field_names=["x", "y", "z", "intensity"], skip_nans=True):
        points.append(p)
    arr = np.array(points, dtype=np.float32)
    np.save(str(save_path), arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_root', type=str, required=True, help='Bag files root directory')
    parser.add_argument('--save_root', type=str, required=True, help='Output root directory')
    parser.add_argument('--date', type=str, required=True, help='Date string (e.g. 2024-06-15)')
    parser.add_argument('--camera_topic', type=str, default='/blackfly/image_raw/compressed')
    parser.add_argument('--odom_topic', type=str, default='/novatel/oem7/odom')
    parser.add_argument('--ouster1_topic', type=str, default='/ouster1/points')
    parser.add_argument('--ouster2_topic', type=str, default='/ouster2/points')
    parser.add_argument('--ouster3_topic', type=str, default='/ouster3/points')
    args = parser.parse_args()

    bag_root = Path(args.bag_root) / args.date
    save_root = Path(args.save_root) / args.date
    camera_dir = save_root / "camera"
    odom_dir = save_root / "odom"
    ouster1_dir = save_root / "ouster1"
    ouster2_dir = save_root / "ouster2"
    ouster3_dir = save_root / "ouster3"
    camera_dir.mkdir(parents=True, exist_ok=True)
    odom_dir.mkdir(parents=True, exist_ok=True)
    ouster1_dir.mkdir(parents=True, exist_ok=True)
    ouster2_dir.mkdir(parents=True, exist_ok=True)
    ouster3_dir.mkdir(parents=True, exist_ok=True)
    skipped_log_path = save_root / "skipped_bags.txt"

    bag_files = sorted(bag_root.rglob('*.bag'))
    print(f"Found {len(bag_files)} bag files")

    for bag_file in bag_files:
        print(f"\n📦 Processing bag file: {bag_file}")
        try:
            with rosbag.Bag(str(bag_file), "r") as bag:
                for topic, msg, t in bag.read_messages(topics=[
                    args.camera_topic, args.odom_topic,
                    args.ouster1_topic, args.ouster2_topic, args.ouster3_topic
                ]):
                    timestamp_str = convert_timestamp(t.to_nsec())
                    if topic == args.camera_topic:
                        save_path = camera_dir / f"{timestamp_str}.png"
                        if not save_path.exists():
                            save_camera(msg, save_path)
                    elif topic == args.odom_topic:
                        save_path = odom_dir / f"{timestamp_str}.json"
                        if not save_path.exists():
                            save_odom(msg, save_path)
                    elif topic == args.ouster1_topic:
                        save_path = ouster1_dir / f"{timestamp_str}.npy"
                        if not save_path.exists():
                            save_pointcloud2(msg, save_path)
                    elif topic == args.ouster2_topic:
                        save_path = ouster2_dir / f"{timestamp_str}.npy"
                        if not save_path.exists():
                            save_pointcloud2(msg, save_path)
                    elif topic == args.ouster3_topic:
                        save_path = ouster3_dir / f"{timestamp_str}.npy"
                        if not save_path.exists():
                            save_pointcloud2(msg, save_path)
        except Exception as e:
            with open(skipped_log_path, 'a') as f:
                f.write(f"{datetime.now(tz=KST).isoformat()} | {bag_file} | {e}\n")

if __name__ == '__main__':
    main()