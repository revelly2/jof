import os
import shutil
from ultralytics import YOLO

def train_yolo():
    """
    Python script to train YOLOv8 locally on your rice disease dataset.
    This automatically exports the trained model to a web-friendly ONNX format
    and copies it directly to the model/best.onnx directory of the web application.
    """
    # 1. Load a pre-trained YOLOv8-nano detection model
    # YOLOv8n is lightweight and optimized for real-time browser CPU execution
    print("Loading pre-trained YOLOv8-nano model weights...")
    model = YOLO("yolov8n.pt")
    
    # Check for GPU acceleration
    import torch
    cuda_available = torch.cuda.is_available()
    device = "0" if cuda_available else "cpu"
    print(f"CUDA GPU Acceleration Available: {cuda_available}")
    print(f"Selected Training Device: {device.upper()}")
    if not cuda_available:
        print("NOTE: Running on CPU. Training will take significant time (~25 mins/epoch).")
        print("To verify the UI flow quickly, you can edit train.py to use epochs=3 and imgsz=320.")
    
    # 2. Train the model on your custom dataset
    print("Starting YOLOv8 training on local dataset config 'data.yaml'...")
    model.train(
        data="data.yaml",   # Path to the dataset yaml configuration file
        epochs=3,           # Reduced epochs for quick verification
        imgsz=320,          # Reduced image size for fast verification
        batch=8,            # Smaller batch size for compatibility
        device=device,      # Set to device detected above
        workers=2           # Number of data loader CPU worker threads
    )
    
    # 3. Export the trained PyTorch weights (.pt) to ONNX format
    print("\nTraining complete! Exporting best weights to web-friendly ONNX format...")
    onnx_path = model.export(format="onnx", imgsz=320)
    print(f"ONNX model exported successfully to: {onnx_path}")
    
    # 4. Copy the exported best.onnx model directly to the web app's model/ directory
    web_model_dir = os.path.abspath("./model")
    os.makedirs(web_model_dir, exist_ok=True)
    web_model_path = os.path.join(web_model_dir, "best.onnx")
    
    shutil.copy(onnx_path, web_model_path)
    print(f"\nSUCCESS: Copied model directly to web application: {web_model_path}")
    print("You can now restart your Node server and run live browser inferences!")

if __name__ == "__main__":
    train_yolo()
