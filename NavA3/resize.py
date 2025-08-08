import os
from PIL import Image

def resize_images(input_dir, output_dir, size=(640, 480)):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            try:
                img = Image.open(in_path)
                img = img.resize(size, Image.BILINEAR)
                img.save(out_path)
                print(f"Resized {fname} -> {out_path}")
            except Exception as e:
                print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    input_dir = "demo_input_lf"
    output_dir = "demo_input_resized"
    resize_images(input_dir, output_dir)