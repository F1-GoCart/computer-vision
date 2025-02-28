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

    START = sv.Point(636, 252)
    END = sv.Point(0, 252)
    line_zone = sv.LineZone(start=START, end=END, triggering_anchors=[Position.CENTER])

    # Initialize annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=0.5)
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator(thickness=2)

    # Load your custom model using Ultralytics and move it to GPU
    model = YOLO("gocart1.pt").to("cuda")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Assume 'frame' is your image read via cv2 (shape: H x W x C)
        # Convert frame from BGR (OpenCV default) to RGB if your model expects RGB:
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Convert NumPy array to a PyTorch tensor (and change shape to [1, C, H, W])
        frame_tensor = torch.from_numpy(frame_rgb).float()  # Convert to float32
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, H, W]

        # Optionally, normalize pixel values to [0,1] if needed:
        frame_tensor /= 255.0

        # Move tensor to GPU:
        frame_tensor = frame_tensor.to('cuda')

        # Run inference using the Ultralytics YOLO model
        with torch.no_grad():
            results = model(frame_tensor, verbose=False)[0]

        # Convert YOLO results to Supervision detections
        detections = sv.Detections.from_ultralytics(results)
        conf_threshold = 0.75
        detections = detections[detections.confidence > conf_threshold]
        tracked_detections = tracker.update_with_detections(detections)



        # Build labels from tracked detections (adjust as needed)
        labels = [
            f"{model.names[class_id]}: {confidence:.2f}"
            for class_id, confidence in zip(tracked_detections.class_id, tracked_detections.confidence)
        ]

        # if len(tracked_detections) > 0:
        #     print("Detections:", labels)

        annotated_frame = box_annotator.annotate(frame.copy(), detections=tracked_detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=tracked_detections, labels=labels)
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame,
            detections=tracked_detections)
        annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
        crossed_in, crossed_out = line_zone.trigger(detections=tracked_detections)

        tasks = []
        for i, (flag_in, flag_out) in enumerate(zip(crossed_in, crossed_out)):
            if flag_in:
                print("Crossed IN:", tracked_detections.class_id[i])
                if tracked_detections.class_id[i] == 0:
                    item = "1"
                elif tracked_detections.class_id[i] == 2:
                    item = "2"
                else:
                    item = "0"
                asyncio.create_task(post_scanned_in(item))
            if flag_out:
                print("Crossed OUT:", tracked_detections.class_id[i])
                if tracked_detections.class_id[i] == 0:
                    item = "1"
                elif tracked_detections.class_id[i] == 2:
                    item = "2"
                else:
                    item = "0"
                asyncio.create_task(post_scanned_out(item))

        if tasks:
            responses = await asyncio.gather(*tasks)
            for r in responses:
                print(r.text)



        cv.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow('frame', annotated_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(0)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())
