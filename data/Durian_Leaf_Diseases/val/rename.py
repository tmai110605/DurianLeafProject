from pathlib import Path
import pandas as pd
from PIL import Image

# Your validation directory
VAL_ROOT = Path(
    r"C:\Users\Lenovo\Downloads\A Durian Leaf Image Dataset of Common Diseases in Vietnam for Agricultural Diagnosis\Durian_Leaf_Diseases\val"
)

# Metadata save path
METADATA_PATH = VAL_ROOT / "metadata_val.csv"

# Class order to traverse
CLASS_ORDER = [
    "algal",
    "allocaridara attack",
    "blight",
    "healthy",
    "phomopsis"
]

# Fixed label_id map
LABEL_MAP = {
    "healthy": 0,
    "algal": 1,
    "allocaridara attack": 2,
    "blight": 3,
    "phomopsis": 4
}

# Standard label names for CSV
NORMALIZED_LABEL = {
    "healthy": "healthy",
    "algal": "algal",
    "allocaridara attack": "allocaridara_attack",
    "blight": "blight",
    "phomopsis": "phomopsis"
}

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]

# Delete existing metadata if present
if METADATA_PATH.exists():
    METADATA_PATH.unlink()

# Step 1: rename all images to temporary names to prevent filename collisions
temp_records = []

for class_name in CLASS_ORDER:
    class_dir = VAL_ROOT / class_name

    if not class_dir.exists():
        print(f"Folder not found: {class_dir}")
        continue

    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(class_dir.glob(f"*{ext}"))
        image_paths.extend(class_dir.glob(f"*{ext.upper()}"))

    image_paths = sorted(set(image_paths))

    print(f"{class_name}: {len(image_paths)} images")

    for i, old_path in enumerate(image_paths, start=1):
        temp_name = f"__tmp_rename_{class_name.replace(' ', '_')}_{i:06d}{old_path.suffix.lower()}"
        temp_path = class_dir / temp_name

        # If temporary files remain from a previous failed run, delete them
        if temp_path.exists():
            temp_path.unlink()

        old_path.rename(temp_path)

        temp_records.append({
            "class_name": class_name,
            "temp_path": temp_path,
            "original_file_name": old_path.name,
            "original_path": str(old_path.relative_to(VAL_ROOT)).replace("\\", "/"),
            "original_extension": old_path.suffix.lower()
        })

# Step 2: convert to JPG and rename with sequential standard IDs
rows = []
counter = 1

for record in temp_records:
    class_name = record["class_name"]
    temp_path = record["temp_path"]

    image_id = f"DLDD_VAL_{counter:06d}"
    new_name = f"{image_id}.jpg"
    new_path = temp_path.parent / new_name

    # If the target name already exists, remove it before overwriting
    if new_path.exists():
        new_path.unlink()

    try:
        with Image.open(temp_path) as img:
            img = img.convert("RGB")
            img.save(new_path, "JPEG", quality=95)

        # Delete temporary image after successful conversion
        temp_path.unlink()

        rows.append({
            "image_id": image_id,
            "file_name": new_name,
            "file_path": str(new_path.relative_to(VAL_ROOT)).replace("\\", "/"),
            "original_file_name": record["original_file_name"],
            "original_path": record["original_path"],
            "original_extension": record["original_extension"],
            "split": "val",
            "label_id": LABEL_MAP[class_name],
            "disease_type": NORMALIZED_LABEL[class_name],
            "verification_status": "verified"
        })

        counter += 1

    except Exception as e:
        print(f"Error converting {temp_path}: {e}")
        # Do not delete temp_path if error occurs for inspection

df = pd.DataFrame(rows)
df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")

print("\nCompleted renaming + JPG conversion for VAL.")
print(f"Total successfully processed images: {len(df)}")
print(f"Metadata saved at: {METADATA_PATH}")

if len(df) > 0:
    print(df.groupby("disease_type").size())
else:
    print("No images were processed.")