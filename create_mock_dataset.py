import os
import struct
import random

def write_bmp(path, width, height, color_func):
    """
    Writes a raw 24-bit BMP image file without using external libraries (Pillow/OpenCV).
    """
    row_size = (width * 3 + 3) & ~3
    pixel_data_size = row_size * height
    file_size = 54 + pixel_data_size
    
    # Pack standard BMP 54-byte header
    header = struct.pack(
        '<2sIHHIiiiHHIIiiII',
        b'BM', file_size, 0, 0, 54, 40, width, height, 1, 24, 0, pixel_data_size, 2835, 2835, 0, 0
    )
    
    with open(path, 'wb') as f:
        f.write(header)
        for y in range(height):
            row = bytearray()
            for x in range(width):
                b, g, r = color_func(x, y)
                row.extend([b, g, r])
            # Pad row to 4-byte boundary
            while len(row) % 4 != 0:
                row.append(0)
            f.write(row)

def generate_synthetic_dataset():
    """
    Generates 2,000 synthetic leaf images (64x64 px) with distinct disease spots and label files.
    - 1,600 images for training (80%)
    - 400 images for validation (20%)
    """
    print("Initializing synthetic agritech dataset builder...")
    
    # Path settings
    base_dir = "./dataset"
    splits = {
        "train": 1600,
        "val": 400
    }
    
    # Disease color specifications
    disease_colors = {
        0: (70, 85, 110),     # Blast: Grayish-brown
        1: (55, 105, 165),    # BPH: Warm orange-brown
        2: (40, 90, 135),     # Stem Borer: Dark yellow-brown
        3: (90, 185, 210),    # BLB: Pale yellow-orange
        4: (205, 225, 215)    # Leaf Folder: Papery white
    }
    
    img_size = 64 # Small size for instant local writing and training
    
    for split_name, count in splits.items():
        print(f"Generating {count} synthetic files for {split_name} split...")
        
        img_dir = os.path.join(base_dir, split_name, "images")
        lbl_dir = os.path.join(base_dir, split_name, "labels")
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        for idx in range(count):
            # Pick a disease class at random (0 to 4)
            class_id = random.randint(0, 4)
            
            # Place a spot randomly on the leaf canvas
            radius = random.randint(4, 12)
            cx = random.randint(radius + 5, img_size - radius - 5)
            cy = random.randint(radius + 5, img_size - radius - 5)
            
            # BPH clusters specifically at the bottom
            if class_id == 1:
                cy = random.randint(img_size - radius - 15, img_size - radius - 5)
            
            # Draw green leaf background with colored disease spot
            def get_pixel_color(x, y):
                dist2 = (x - cx)**2 + (y - cy)**2
                if dist2 < radius**2:
                    return disease_colors[class_id]
                # Default leaf green channel variation
                green_val = 140 + int(20 * random.random())
                return (35, green_val, 45)
            
            # File paths
            file_base = f"crop_{split_name}_{idx}"
            img_path = os.path.join(img_dir, f"{file_base}.bmp")
            lbl_path = os.path.join(lbl_dir, f"{file_base}.txt")
            
            # Write BMP image
            write_bmp(img_path, img_size, img_size, get_pixel_color)
            
            # Write normalized YOLO label txt file: class_id cx cy w h
            cx_norm = cx / img_size
            cy_norm = cy / img_size
            w_norm = (radius * 2) / img_size
            h_norm = (radius * 2) / img_size
            
            with open(lbl_path, "w") as f:
                f.write(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                
    print("\nSUCCESS: Created 2,000 synthetic leaf images and YOLO label txt coordinates!")
    print(f"Location: {os.path.abspath(base_dir)}")
    print("You can now run 'python train.py' to verify your YOLOv8 local training works!")

if __name__ == "__main__":
    generate_synthetic_dataset()
