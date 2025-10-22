import os
import cv2
import glob
import argparse

def yolo_to_bbox(yolo_bbox, img_width, img_height):
    class_id, x_center, y_center, width, height = map(float, yolo_bbox)
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    
    return int(class_id), x_min, y_min, x_max, y_max

def crop_bounding_boxes(images_path, labels_path, output_path, class_ids):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    image_files = glob.glob(os.path.join(images_path, "*.jpg"))  # Adjust for other formats if needed
    
    for image_file in image_files:
        img = cv2.imread(image_file)
        img_height, img_width, _ = img.shape
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(labels_path, base_name + ".txt")
        
        if not os.path.exists(label_file):
            continue
        
        with open(label_file, "r") as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            bbox = line.strip().split()
            class_id, x_min, y_min, x_max, y_max = yolo_to_bbox(bbox, img_width, img_height)
            
            if class_ids and class_id not in class_ids:
                continue
            
            cropped_img = img[y_min:y_max, x_min:x_max]
            
            if cropped_img.size == 0:
                continue
            
            output_file = os.path.join(output_path, f"{base_name}_{i}_{int(class_id)}.jpg")
            cv2.imwrite(output_file, cropped_img)
            print(f"Saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop bounding boxes from YOLO format dataset.")
    parser.add_argument("--images_dir", type=str, help="Path to the directory containing images.")
    parser.add_argument("--labels_dir", type=str, help="Path to the directory containing YOLO annotations.")
    parser.add_argument("--output_dir", type=str, help="Path to the directory where cropped images will be saved.")
    parser.add_argument("--class_ids", type=int, nargs="*", default=None, help="List of class IDs to crop (optional). If not provided, all classes will be cropped.")
    
    args = parser.parse_args()
    crop_bounding_boxes(args.images_dir, args.labels_dir, args.output_dir, set(args.class_ids) if args.class_ids else None)
