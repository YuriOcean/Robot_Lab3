#!/usr/bin/env python3
"""Persist recognition, pick, place and exception results for each sorting run."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SortingResultLoggerNode(Node):
    """Build per-run JSON/CSV reports from the existing sorting status topics."""

    SOFTWARE_VERSION = "2026-09-14-bin-slots-v5"
    TERMINAL_EVENTS = {"succeeded", "failed", "rejected", "stopped"}

    def __init__(self) -> None:
        super().__init__("sorting_result_logger")

        self.declare_parameter("sorting_status_topic", "/sorting/status")
        self.declare_parameter("arm_status_topic", "/arm_task/status")
        self.declare_parameter("chassis_status_topic", "/chassis_motion/status")
        self.declare_parameter("grid_state_topic", "/grid_states")
        self.declare_parameter(
            "logger_status_topic", "/sorting_result_logger/status"
        )
        self.declare_parameter(
            "log_directory", "~/experiment3_ws/result_logs"
        )
        self.declare_parameter(
            "task_order", ["G4", "G1", "G5", "G2", "G6", "G3"]
        )

        sorting_topic = self._string_parameter("sorting_status_topic")
        arm_topic = self._string_parameter("arm_status_topic")
        chassis_topic = self._string_parameter("chassis_status_topic")
        grid_topic = self._string_parameter("grid_state_topic")
        logger_topic = self._string_parameter("logger_status_topic")

        configured_directory = self._string_parameter("log_directory")
        self.log_directory = Path(
            os.path.expandvars(os.path.expanduser(configured_directory))
        ).resolve()
        self.log_directory.mkdir(parents=True, exist_ok=True)

        self.task_order = [
            str(value) for value in self.get_parameter("task_order").value
        ]

        self.status_publisher = self.create_publisher(String, logger_topic, 10)
        self.create_subscription(
            String, sorting_topic, self._sorting_status_callback, 50
        )
        self.create_subscription(String, arm_topic, self._arm_status_callback, 50)
        self.create_subscription(
            String, chassis_topic, self._chassis_status_callback, 50
        )
        self.create_subscription(String, grid_topic, self._grid_state_callback, 10)

        self.current_run: Optional[int] = None
        self.report: Optional[Dict[str, Any]] = None
        self.latest_grid_state: Dict[str, Any] = {}
        self.command_context: Dict[str, Dict[str, Any]] = {}
        self.pending_arm_events: Dict[str, List[Dict[str, Any]]] = {}

        self.json_path: Optional[Path] = None
        self.csv_path: Optional[Path] = None
        self.events_path: Optional[Path] = None
        self.write_error_reported = False

        self.get_logger().info(
            f"Sorting result logger {self.SOFTWARE_VERSION} ready: "
            f"{self.log_directory}"
        )
        self.get_logger().info("Task order: " + " -> ".join(self.task_order))
        self._publish_logger_status("ready", True, "Waiting for a sorting run")

    def _string_parameter(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    @staticmethod
    def _safe_json(message: String) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _empty_operation() -> Dict[str, Any]:
        return {
            "status": "not_started",
            "success": None,
            "command_id": "",
            "started_at": "",
            "finished_at": "",
            "message": "",
            "holding_state": "",
        }

    def _new_object_record(self, grid_id: str, order: int) -> Dict[str, Any]:
        return {
            "grid_id": grid_id,
            "order": order,
            "recognized_class": "pending",
            "recognition_status": "pending",
            "recognition_time": "",
            "pick": self._empty_operation(),
            "place": self._empty_operation(),
            "placement_target": {
                "bin": "",
                "slot": None,
                "chassis_x": None,
                "chassis_y": None,
            },
            "final_result": "pending",
            "exceptions": [],
        }

    def _start_report(self, run_number: int, object_limit: int) -> None:
        now = datetime.now().astimezone()
        file_stamp = now.strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"sorting_run_{run_number:03d}_{file_stamp}"

        self.current_run = run_number
        self.json_path = self.log_directory / f"{base_name}.json"
        self.csv_path = self.log_directory / f"{base_name}.csv"
        self.events_path = self.log_directory / f"{base_name}_events.jsonl"
        self.command_context = {}
        self.pending_arm_events = {}
        self.write_error_reported = False

        required_order = self.task_order[:object_limit]
        self.report = {
            "schema_version": 2,
            "run": run_number,
            "status": "running",
            "started_at": now.isoformat(timespec="milliseconds"),
            "finished_at": "",
            "object_limit": object_limit,
            "task_order": required_order,
            "last_event": "",
            "last_message": "",
            "recognition_snapshot": {},
            "objects": {
                grid_id: self._new_object_record(grid_id, index + 1)
                for index, grid_id in enumerate(required_order)
            },
            "exceptions": [],
            "files": {
                "json": str(self.json_path),
                "csv": str(self.csv_path),
                "events": str(self.events_path),
            },
        }

        self._append_event(
            "logger",
            {
                "event": "run_report_started",
                "run": run_number,
                "object_limit": object_limit,
            },
        )
        self._write_report()
        self.get_logger().info(
            f"Run {run_number} report started: {self.json_path}"
        )
        self._publish_logger_status(
            "report_started", True, f"Recording run {run_number}"
        )

    def _ensure_report(self, payload: Dict[str, Any]) -> bool:
        try:
            run_number = int(payload.get("run", 0))
            object_limit = int(payload.get("object_limit", 6))
        except (TypeError, ValueError):
            return False

        if run_number <= 0:
            return False
        object_limit = max(1, min(object_limit, len(self.task_order)))

        if self.report is None or self.current_run != run_number:
            self._start_report(run_number, object_limit)
        return True

    def _grid_state_callback(self, message: String) -> None:
        payload = self._safe_json(message)
        if payload is not None:
            self.latest_grid_state = payload

    def _sorting_status_callback(self, message: String) -> None:
        payload = self._safe_json(message)
        if payload is None or not self._ensure_report(payload):
            return

        assert self.report is not None
        event = str(payload.get("event", ""))
        current_grid = str(payload.get("current_grid", ""))
        current_class = str(payload.get("current_class", ""))
        command_id = str(payload.get("active_command_id", ""))
        active_child = str(payload.get("active_child", ""))
        message_text = str(payload.get("message", ""))

        self._append_event("sorting_coordinator", payload)
        self.report["last_event"] = event
        self.report["last_message"] = message_text

        if event == "snapshot_frozen":
            snapshot = payload.get("snapshot", {})
            if isinstance(snapshot, dict):
                self.report["recognition_snapshot"] = dict(snapshot)
                for grid_id, record in self.report["objects"].items():
                    label = str(snapshot.get(grid_id, "empty")).strip().lower()
                    record["recognized_class"] = label
                    record["recognition_status"] = (
                        "recognized" if label in ("bottle", "apple") else "skipped"
                    )
                    record["recognition_time"] = self._timestamp()

        if event == "primitive_started" and command_id:
            slot_position = payload.get("target_slot_position")
            self.command_context[command_id] = {
                "grid_id": current_grid,
                "object_class": current_class,
                "child": active_child,
                "description": message_text,
                "target_bin": str(payload.get("target_bin", "")),
                "target_slot": payload.get("target_slot"),
                "target_slot_position": slot_position,
            }
            self._update_placement_target(current_grid, payload)
            self._consume_pending_arm_events(command_id)

        if event in ("grid_completed", "grid_skipped"):
            record = self._object_record(current_grid)
            if record is not None:
                self._update_placement_target(current_grid, payload)
                if current_class:
                    record["recognized_class"] = current_class
                record["final_result"] = (
                    "completed" if event == "grid_completed" else "skipped"
                )

        if event == "failed":
            self._record_exception(
                source="sorting_coordinator",
                message=message_text,
                grid_id=current_grid,
                command_id=command_id,
            )
            self.report["status"] = "failed"
            self.report["finished_at"] = self._timestamp()
        elif event == "stopped":
            self._record_exception(
                source="operator_or_coordinator_stop",
                message=message_text,
                grid_id=current_grid,
                command_id=command_id,
            )
            self.report["status"] = "stopped"
            self.report["finished_at"] = self._timestamp()
        elif event == "completed":
            self.report["status"] = "completed"
            self.report["finished_at"] = self._timestamp()

        self._write_report()

        if event in ("completed", "failed", "stopped"):
            self.get_logger().info(
                f"Run {self.current_run} finalized as {self.report['status']}: "
                f"{self.json_path}"
            )
            self._publish_logger_status(
                "report_finalized",
                event == "completed",
                f"Run {self.current_run}: {self.report['status']}",
            )

    def _arm_status_callback(self, message: String) -> None:
        payload = self._safe_json(message)
        if payload is None or self.report is None:
            return

        self._append_event("arm_task", payload)
        command_id = str(payload.get("id", ""))
        if not command_id:
            return

        if command_id not in self.command_context:
            self.pending_arm_events.setdefault(command_id, []).append(payload)
            return

        self._apply_arm_event(payload)

    def _consume_pending_arm_events(self, command_id: str) -> None:
        for payload in self.pending_arm_events.pop(command_id, []):
            self._apply_arm_event(payload)

    def _apply_arm_event(self, payload: Dict[str, Any]) -> None:
        assert self.report is not None
        command_id = str(payload.get("id", ""))
        context = self.command_context.get(command_id)
        if context is None:
            return

        task = str(payload.get("task", "")).strip().lower()
        if task not in ("pick", "place"):
            description = context.get("description", "").lower()
            if description.startswith("pick "):
                task = "pick"
            elif description.startswith("place "):
                task = "place"
            else:
                return

        grid_id = context.get("grid_id", "")
        record = self._object_record(grid_id)
        if record is None:
            return

        operation = record[task]
        event = str(payload.get("event", "")).strip().lower()
        success = bool(payload.get("success", False))
        message_text = str(payload.get("message", ""))

        operation["command_id"] = command_id
        operation["message"] = message_text
        operation["holding_state"] = str(payload.get("holding_state", ""))

        if not operation["started_at"]:
            operation["started_at"] = self._timestamp()

        if event in self.TERMINAL_EVENTS:
            operation["status"] = event
            operation["success"] = success
            operation["finished_at"] = self._timestamp()

            if event != "succeeded" or not success:
                record["final_result"] = f"{task}_failed"
                self._record_exception(
                    source="arm_task",
                    message=message_text or f"{task} returned {event}",
                    grid_id=grid_id,
                    command_id=command_id,
                    stage=task,
                )
        else:
            operation["status"] = "running"

        self._write_report()

    def _chassis_status_callback(self, message: String) -> None:
        payload = self._safe_json(message)
        if payload is None or self.report is None:
            return

        self._append_event("chassis_motion", payload)
        event = str(payload.get("event", "")).strip().lower()
        success = bool(payload.get("success", False))
        if event in ("failed", "rejected") or (
            event in self.TERMINAL_EVENTS and not success
        ):
            command_id = str(payload.get("id", ""))
            context = self.command_context.get(command_id, {})
            self._record_exception(
                source="chassis_motion",
                message=str(payload.get("message", "chassis motion failed")),
                grid_id=context.get("grid_id", ""),
                command_id=command_id,
                stage="chassis_motion",
            )
            self._write_report()

    def _object_record(self, grid_id: str) -> Optional[Dict[str, Any]]:
        if self.report is None:
            return None
        objects = self.report.get("objects", {})
        record = objects.get(grid_id)
        return record if isinstance(record, dict) else None

    def _update_placement_target(
        self, grid_id: str, payload: Dict[str, Any]
    ) -> None:
        record = self._object_record(grid_id)
        if record is None:
            return
        target_bin = str(payload.get("target_bin", ""))
        target_slot = payload.get("target_slot")
        position = payload.get("target_slot_position")
        if not target_bin or target_slot is None:
            return
        target = record["placement_target"]
        target["bin"] = target_bin
        target["slot"] = target_slot
        if isinstance(position, dict):
            target["chassis_x"] = position.get("x")
            target["chassis_y"] = position.get("y")

    def _record_exception(
        self,
        source: str,
        message: str,
        grid_id: str = "",
        command_id: str = "",
        stage: str = "",
    ) -> None:
        if self.report is None:
            return

        exception = {
            "time": self._timestamp(),
            "source": source,
            "grid_id": grid_id,
            "stage": stage,
            "command_id": command_id,
            "message": message,
        }
        self.report["exceptions"].append(exception)

        record = self._object_record(grid_id)
        if record is not None:
            record["exceptions"].append(exception)

    def _append_event(self, source: str, payload: Dict[str, Any]) -> None:
        if self.events_path is None:
            return
        event_record = {
            "received_at": self._timestamp(),
            "source": source,
            "payload": payload,
        }
        try:
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event_record, ensure_ascii=False) + "\n")
        except OSError as exception:
            self._report_write_error(exception)

    def _write_report(self) -> None:
        if self.report is None or self.json_path is None or self.csv_path is None:
            return
        try:
            json_temporary = self.json_path.with_suffix(".json.tmp")
            json_temporary.write_text(
                json.dumps(self.report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(json_temporary, self.json_path)

            csv_temporary = self.csv_path.with_suffix(".csv.tmp")
            with csv_temporary.open("w", encoding="utf-8-sig", newline="") as stream:
                fieldnames = [
                    "run",
                    "order",
                    "grid_id",
                    "recognized_class",
                    "recognition_status",
                    "pick_status",
                    "pick_success",
                    "pick_message",
                    "place_status",
                    "place_success",
                    "place_message",
                    "target_bin",
                    "target_slot",
                    "target_chassis_x",
                    "target_chassis_y",
                    "final_result",
                    "exception_info",
                ]
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for record in self.report["objects"].values():
                    writer.writerow(
                        {
                            "run": self.report["run"],
                            "order": record["order"],
                            "grid_id": record["grid_id"],
                            "recognized_class": record["recognized_class"],
                            "recognition_status": record["recognition_status"],
                            "pick_status": record["pick"]["status"],
                            "pick_success": record["pick"]["success"],
                            "pick_message": record["pick"]["message"],
                            "place_status": record["place"]["status"],
                            "place_success": record["place"]["success"],
                            "place_message": record["place"]["message"],
                            "target_bin": record["placement_target"]["bin"],
                            "target_slot": record["placement_target"]["slot"],
                            "target_chassis_x": record["placement_target"]["chassis_x"],
                            "target_chassis_y": record["placement_target"]["chassis_y"],
                            "final_result": record["final_result"],
                            "exception_info": " | ".join(
                                item["message"] for item in record["exceptions"]
                            ),
                        }
                    )
            os.replace(csv_temporary, self.csv_path)
        except OSError as exception:
            self._report_write_error(exception)

    def _report_write_error(self, exception: OSError) -> None:
        if not self.write_error_reported:
            self.write_error_reported = True
            self.get_logger().error(f"Result file write failed: {exception}")
            self._publish_logger_status("write_failed", False, str(exception))

    def _publish_logger_status(
        self, event: str, success: bool, message: str
    ) -> None:
        payload = {
            "event": event,
            "success": success,
            "run": self.current_run,
            "message": message,
            "json_file": str(self.json_path) if self.json_path else "",
            "csv_file": str(self.csv_path) if self.csv_path else "",
            "events_file": str(self.events_path) if self.events_path else "",
        }
        output = String()
        output.data = json.dumps(payload, ensure_ascii=False)
        self.status_publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SortingResultLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

