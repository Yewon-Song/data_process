#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry   # just for type hints

### ─── CONFIG ────────────────────────────────────────────────────────────────
BAG_FILE   = "/mnt/sda/2024-06-15/2024-06-15-11-54-13_2.bag"
DATE_STR   = "2024-06-15"            # 폴더 이름에만 쓰임
SAVE_ROOT  = Path("/mnt/sda/bag_per_frame")  # 결과 저장 루트

TOPICS = {
    "camera" : "/blackfly/image_raw/compressed",
    "lidar"  : "/ouster2/points",
    "odom"   : "/novatel/oem7/odom",
}
##############################################################################

KST = timezone(timedelta(hours=9))


def ts_to_str(nsec: int) -> str:
    """ROS Time (ns) → 'YYYY-MM-DD_HH-MM-SS.xxxx' (0.1 ms)"""
    t = datetime.fromtimestamp(nsec / 1e9, tz=KST)
    return t.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-4]


def odom_to_dict(msg: Odometry) -> dict:
    """nav_msgs/Odometry → 파이썬 dict (JSON 직렬화용)"""
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


def prep_dirs():
    """pre-make sensor-specific folders so we don’t race later."""
    for key in ["camera", "lidar", "odom"]:
        (SAVE_ROOT / DATE_STR / key).mkdir(parents=True, exist_ok=True)


def main():
    prep_dirs()

    print(f"🔥  Crunching {Path(BAG_FILE).name} ...")
    with rosbag.Bag(BAG_FILE, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=list(TOPICS.values())):
            ts = ts_to_str(t.to_nsec())

            # ── CAMERA ────────────────────────────────────────────────────────
            if topic == TOPICS["camera"]:
                raw = np.frombuffer(msg.data, np.uint8)
                bayer = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)   # still Bayer-RG
                rgb   = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2RGB)
                out   = SAVE_ROOT / DATE_STR / "camera" / f"{ts}.png"
                cv2.imwrite(out.as_posix(), rgb)

            # ── LIDAR (Ouster2) ──────────────────────────────────────────────
            elif topic == TOPICS["lidar"]:
                pts = np.asarray(
                    list(pc2.read_points(msg,
                                         field_names=("x", "y", "z", "intensity"),
                                         skip_nans=True))
                )
                out = SAVE_ROOT / DATE_STR / "lidar" / f"{ts}.npy"
                np.save(out.as_posix(), pts)

            # ── ODOM (NovAtel) ───────────────────────────────────────────────
            elif topic == TOPICS["odom"]:
                out = SAVE_ROOT / DATE_STR / "odom" / f"{ts}.json"
                with open(out, "w") as f:
                    json.dump(odom_to_dict(msg), f, indent=4)

    print("✅  Done, all goodies are under:", SAVE_ROOT / DATE_STR)


if __name__ == "__main__":
    main()
