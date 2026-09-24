#!/usr/bin/env python3
"""ROS 2 YOLO detector for bottle/apple using the RoboMaster EP camera."""

import os
import time
from typing import Dict, List

import cv2
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class YoloDetectorNode(Node):
    """Detect selected YOLO classes in a ROS image stream."""

    def __init__(self) -> None:
        super().__init__('yolo_detector')

        default_model = os.path.join(
            get_package_share_directory('ep_object_sorting'),
            'models',
            'yolo11n.pt',
        )

        self.declare_parameter('input_topic', '/camera/image_color')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('annotated_topic', '/yolo/annotated_image')
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('target_classes', ['bottle', 'apple'])
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('max_inference_hz', 5.0)
        self.declare_parameter('publish_annotated_image', True)

        input_topic = str(self.get_parameter('input_topic').value)
        detections_topic = str(self.get_parameter('detections_topic').value)
        annotated_topic = str(self.get_parameter('annotated_topic').value)
        model_path = str(self.get_parameter('model_path').value)
        target_names = list(self.get_parameter('target_classes').value)

        self.confidence_threshold = float(
            self.get_parameter('confidence_threshold').value
        )
        self.image_size = int(self.get_parameter('image_size').value)
        self.device = str(self.get_parameter('device').value)
        self.max_inference_hz = float(
            self.get_parameter('max_inference_hz').value
        )
        self.publish_annotated_image = bool(
            self.get_parameter('publish_annotated_image').value
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f'YOLO model does not exist: {model_path}. '
                'Copy yolo11n.pt into the package models directory or set model_path.'
            )

        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)

        self.class_name_by_id: Dict[int, str] = {
            int(class_id): str(class_name)
            for class_id, class_name in self.model.names.items()
        }
        self.target_class_ids: List[int] = [
            class_id
            for class_id, class_name in self.class_name_by_id.items()
            if class_name in target_names
        ]

        found_names = {
            self.class_name_by_id[class_id] for class_id in self.target_class_ids
        }
        missing_names = sorted(set(target_names) - found_names)
        if missing_names:
            raise RuntimeError(
                'Classes are missing from the model: ' + ', '.join(missing_names)
            )

        self.bridge = CvBridge()
        self.last_inference_time = 0.0
        self.frame_counter = 0

        self.detections_publisher = self.create_publisher(
            Detection2DArray,
            detections_topic,
            10,
        )
        self.annotated_publisher = self.create_publisher(
            Image,
            annotated_topic,
            10,
        )
        self.image_subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        target_description = ', '.join(
            f'{self.class_name_by_id[class_id]}({class_id})'
            for class_id in self.target_class_ids
        )
        self.get_logger().info(f'Target classes: {target_description}')
        self.get_logger().info(f'Subscribing to: {input_topic}')
        self.get_logger().info(f'Publishing detections: {detections_topic}')
        if self.publish_annotated_image:
            self.get_logger().info(f'Publishing annotated image: {annotated_topic}')

    def image_callback(self, image_message: Image) -> None:
        """Run inference on the newest permitted camera frame."""
        now = time.monotonic()
        if self.max_inference_hz > 0.0:
            minimum_interval = 1.0 / self.max_inference_hz
            if now - self.last_inference_time < minimum_interval:
                return
        self.last_inference_time = now

        try:
            frame = self.bridge.imgmsg_to_cv2(
                image_message,
                desired_encoding='bgr8',
            )
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                device=self.device,
                classes=self.target_class_ids,
                verbose=False,
            )
        except Exception as exception:  # Keep the camera subscription alive.
            self.get_logger().error(f'YOLO inference failed: {exception}')
            return

        detections_message = Detection2DArray()
        detections_message.header = image_message.header

        result = results[0]
        if result.boxes is not None:
            for detection_index, box in enumerate(result.boxes):
                x_min, y_min, x_max, y_max = [
                    float(value) for value in box.xyxy[0].cpu().tolist()
                ]
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = self.class_name_by_id[class_id]

                detection = Detection2D()
                detection.header = image_message.header
                detection.id = f'{class_name}_{detection_index}'

                center_x = (x_min + x_max) / 2.0
                center_y = (y_min + y_max) / 2.0
                self.set_bbox_center(detection, center_x, center_y)
                detection.bbox.size_x = x_max - x_min
                detection.bbox.size_y = y_max - y_min

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = confidence
                detection.results.append(hypothesis)
                detections_message.detections.append(detection)

        self.detections_publisher.publish(detections_message)

        if self.publish_annotated_image:
            annotated_frame = result.plot()
            annotated_message = self.bridge.cv2_to_imgmsg(
                annotated_frame,
                encoding='bgr8',
            )
            annotated_message.header = image_message.header
            self.annotated_publisher.publish(annotated_message)

        self.frame_counter += 1
        if self.frame_counter % 10 == 0 or detections_message.detections:
            labels = [
                detection.results[0].hypothesis.class_id
                for detection in detections_message.detections
            ]
            summary = ', '.join(labels) if labels else 'none'
            self.get_logger().info(
                f'Detections={len(labels)} [{summary}]'
            )

    @staticmethod
    def set_bbox_center(
        detection: Detection2D,
        center_x: float,
        center_y: float,
    ) -> None:
        """Support the BoundingBox2D center layout used by Humble variants."""
        center = detection.bbox.center
        if hasattr(center, 'position'):
            center.position.x = center_x
            center.position.y = center_y
            if hasattr(center, 'theta'):
                center.theta = 0.0
        else:
            center.x = center_x
            center.y = center_y
            center.theta = 0.0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = YoloDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exception:
        if node is not None:
            node.get_logger().fatal(str(exception))
        else:
            print(f'[yolo_detector] fatal: {exception}')
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

