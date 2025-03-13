import asyncio
import time

import httpx
import torch
from supervision.geometry.core import Position
import cv2 as cv
import supervision as sv
from ultralytics import YOLO
import requests

tracker = sv.ByteTrack()

async def post_scanned_in(item: str):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/scanned-in", json={"class_id": item})
        return response

async def post_scanned_out(item: str):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/remove-item", json={"class_id": item})
        return response


async def main():
    cap = cv.VideoCapture(2)
    frame_width, frame_height = 640, 480
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

    START, END = sv.Point(636, 252), sv.Point(0, 252)
    line_zone = sv.LineZone(
        start=START,
        end=END,
        triggering_anchors=[Position.CENTER],
    )

    annotators = {
        "box": sv.BoxAnnotator(thickness=2),
        "label": sv.LabelAnnotator(),
        "line": sv.LineZoneAnnotator(thickness=3, text_scale=0.5),
    }

    CLASS_ID_MAPPING = {0: "1", 2: "2"}

    model = YOLO("gocart1.pt").to("cuda")
    tracker = sv.ByteTrack()

    async with httpx.AsyncClient() as client:
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            with torch.no_grad():
                results = model(frame_rgb, verbose=False)[0]

            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.confidence > 0.40]
            tracked_detections = tracker.update_with_detections(detections)

            labels = [
                f"{model.names[cid]}: {conf:.2f}"
                for cid, conf in zip(tracked_detections.class_id, tracked_detections.confidence)
            ]

            annotated_frame = annotators["box"].annotate(frame.copy(), tracked_detections)
            annotated_frame = annotators["label"].annotate(annotated_frame, tracked_detections, labels=labels)
            annotated_frame = annotators["line"].annotate(annotated_frame, line_counter=line_zone)

            crossed_in, crossed_out = line_zone.trigger(tracked_detections)
            tasks = []

            for i, (flag_in, flag_out) in enumerate(zip(crossed_in, crossed_out)):
                item = CLASS_ID_MAPPING.get(tracked_detections.class_id[i], "0")

                if flag_in:
                    print("Crossed IN:", item)
                    tasks.append(asyncio.create_task(post_scanned_in(client, item)))

                if flag_out:
                    print("Crossed OUT:", item)
                    tasks.append(asyncio.create_task(post_scanned_out(client, item)))

            if tasks:
                responses = await asyncio.gather(*tasks)
                for r in responses:
                    print(r.text)

            # cv.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
            #            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('frame', annotated_frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
