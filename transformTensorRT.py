from ultralytics import YOLO
#Load your models, ex;
model_n = YOLO('yolov8n.pt')
model_x = YOLO('yolov8x.pt')

print("Exporting YOLOv8n to TensorRT...")
# Export with FP16 for good speedup, ensure GPU is available
# You can specify workspace size, dynamic axes, etc.
# The first export might take a while as TensorRT optimizes.
model_n.export(format='engine', half=True, device=0) # device=0 for first GPU
print("YOLOv8n exported to yolov8n.engine")

print("Exporting YOLOv8x to TensorRT...")
model_x.export(format='engine', half=True, device=0)
print("YOLOv8x exported to yolov8x.engine")

# The exported files will be yolov8n.engine and yolov8x.engine
# You might want to specify output names:
# model_n.export(format='engine', half=True, device=0, imgsz=640, name='yolov8n_fp16_640.engine')