from ultralytics import YOLO # type: ignore

model = YOLO('yolo26n-pose.yaml')
model = YOLO('yolo26n-pose.pt')
model = YOLO('yolo26n-pose.yaml').load('yolo26n-pose.pt')

res = model.train(data = 'coco8-pose.yaml',epochs = 100, imgsz=640)