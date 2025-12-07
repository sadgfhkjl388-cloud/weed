#!/usr/bin/env python3
"""
Visualize Test Predictions on Images

Draw bounding boxes from submission.csv directly on test images and save them.
No 3LC required - just pure image visualization.

Usage:
    python visualize_test_predictions.py

Output:
    - test_predictions_visualized/ directory with annotated images
    - Each image shows all predicted bounding boxes with confidence scores
"""

import pandas as pd
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

SUBMISSION_CSV = "submission.csv"
TEST_IMAGE_DIR = Path("test/images")
OUTPUT_DIR = Path("test_predictions_visualized")

# Class names and colors
CLASS_NAMES = {
    0: "Carpetweed",
    1: "Morning Glory",
    2: "Palmer Amaranth"
}

CLASS_COLORS = {
    0: (0, 255, 0),      # Green
    1: (255, 0, 0),      # Blue
    2: (0, 0, 255)       # Red
}

# Visualization settings
CONFIDENCE_THRESHOLD = 0.25  # Show all predictions (set to 0.25 to filter low confidence)
BOX_THICKNESS = 2
FONT_SCALE = 0.5
FONT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def draw_bbox(image, class_id, confidence, x_center, y_center, width, height):
    """Draw a single bounding box on the image"""

    img_height, img_width = image.shape[:2]

    # Convert normalized YOLO coordinates to pixel coordinates
    x_center_px = int(x_center * img_width)
    y_center_px = int(y_center * img_height)
    box_width_px = int(width * img_width)
    box_height_px = int(height * img_height)

    # Calculate top-left corner
    x1 = int(x_center_px - box_width_px / 2)
    y1 = int(y_center_px - box_height_px / 2)
    x2 = int(x_center_px + box_width_px / 2)
    y2 = int(y_center_px + box_height_px / 2)

    # Get class info
    class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
    color = CLASS_COLORS.get(class_id, (255, 255, 255))

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    # Create label with class name and confidence
    label = f"{class_name} {confidence:.2f}"

    # Get label size for background
    (label_width, label_height), baseline = cv2.getTextSize(
        label, FONT, FONT_SCALE, FONT_THICKNESS
    )

    # Draw label background
    label_y1 = max(y1 - label_height - baseline - 5, 0)
    label_y2 = y1
    cv2.rectangle(
        image,
        (x1, label_y1),
        (x1 + label_width + 5, label_y2),
        color,
        -1  # Filled
    )

    # Draw label text
    cv2.putText(
        image,
        label,
        (x1 + 2, y1 - 5),
        FONT,
        FONT_SCALE,
        (255, 255, 255),  # White text
        FONT_THICKNESS,
        cv2.LINE_AA
    )

    return image

def add_summary_text(image, total_boxes, high_conf_boxes, low_conf_boxes):
    """Add summary statistics to the top of the image"""

    summary_lines = [
        f"Total boxes: {total_boxes}",
        f"High conf (>=0.5): {high_conf_boxes}",
        f"Low conf (<0.1): {low_conf_boxes}"
    ]

    # Draw black background for summary
    bg_height = 80
    cv2.rectangle(image, (0, 0), (image.shape[1], bg_height), (0, 0, 0), -1)

    # Draw summary text
    y_offset = 20
    for line in summary_lines:
        cv2.putText(
            image,
            line,
            (10, y_offset),
            FONT,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        y_offset += 25

    return image

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def visualize_predictions():
    """Visualize all test predictions on images"""

    print("=" * 80)
    print("VISUALIZE TEST PREDICTIONS")
    print("=" * 80)

    # Load submission CSV
    print(f"\n📄 Loading predictions from: {SUBMISSION_CSV}")
    if not Path(SUBMISSION_CSV).exists():
        print(f"\n❌ Error: {SUBMISSION_CSV} not found!")
        print("   Please run 'python predict.py' first.")
        return

    df = pd.read_csv(SUBMISSION_CSV)
    print(f"   ✓ Loaded {len(df)} predictions")

    # Verify test images exist
    print(f"\n📁 Checking test images...")
    if not TEST_IMAGE_DIR.exists():
        print(f"\n❌ Error: {TEST_IMAGE_DIR} not found!")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Output directory: {OUTPUT_DIR}")

    # Statistics
    stats = {
        'total_images': 0,
        'images_with_boxes': 0,
        'images_without_boxes': 0,
        'total_boxes': 0,
        'high_conf_boxes': 0,  # >= 0.5
        'low_conf_boxes': 0,   # < 0.1
        'filtered_boxes': 0,   # Below confidence threshold
    }

    # Process each image
    print(f"\n🎨 Drawing bounding boxes on images...")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        image_id = row['image_id']
        pred_string = row['prediction_string']

        # Load image
        image_path = TEST_IMAGE_DIR / f"{image_id}.jpg"
        if not image_path.exists():
            print(f"\n⚠️  Warning: Image not found: {image_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"\n⚠️  Warning: Failed to load: {image_path}")
            continue

        stats['total_images'] += 1

        # Handle "no box" case
        if pred_string == "no box":
            stats['images_without_boxes'] += 1
            # Add summary text
            image = add_summary_text(image, 0, 0, 0)
        else:
            # Parse predictions
            values = pred_string.split()

            box_count = 0
            high_conf = 0
            low_conf = 0
            filtered = 0

            # Draw each bounding box
            for i in range(0, len(values), 6):
                if i + 5 >= len(values):
                    break

                try:
                    class_id = int(values[i])
                    confidence = float(values[i + 1])
                    x_center = float(values[i + 2])
                    y_center = float(values[i + 3])
                    width = float(values[i + 4])
                    height = float(values[i + 5])

                    stats['total_boxes'] += 1

                    # Track confidence statistics
                    if confidence >= 0.5:
                        high_conf += 1
                        stats['high_conf_boxes'] += 1
                    if confidence < 0.1:
                        low_conf += 1
                        stats['low_conf_boxes'] += 1

                    # Filter by confidence threshold
                    if confidence < CONFIDENCE_THRESHOLD:
                        filtered += 1
                        stats['filtered_boxes'] += 1
                        continue

                    box_count += 1

                    # Draw bounding box
                    image = draw_bbox(
                        image, class_id, confidence,
                        x_center, y_center, width, height
                    )

                except (ValueError, IndexError) as e:
                    continue

            if box_count > 0:
                stats['images_with_boxes'] += 1
            else:
                stats['images_without_boxes'] += 1

            # Add summary text to image
            image = add_summary_text(image, box_count, high_conf, low_conf)

        # Save annotated image
        output_path = OUTPUT_DIR / f"{image_id}_annotated.jpg"
        cv2.imwrite(str(output_path), image)

    # Print final statistics
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)

    print(f"\n📊 Statistics:")
    print(f"  Total images processed: {stats['total_images']}")
    print(f"  Images with predictions: {stats['images_with_boxes']}")
    print(f"  Images without predictions: {stats['images_without_boxes']}")

    print(f"\n📦 Bounding Boxes:")
    print(f"  Total boxes: {stats['total_boxes']}")
    print(f"  High confidence (>=0.5): {stats['high_conf_boxes']} ({stats['high_conf_boxes']/max(stats['total_boxes'],1)*100:.1f}%)")
    print(f"  Low confidence (<0.1): {stats['low_conf_boxes']} ({stats['low_conf_boxes']/max(stats['total_boxes'],1)*100:.1f}%)")

    if CONFIDENCE_THRESHOLD > 0:
        print(f"  Filtered (below {CONFIDENCE_THRESHOLD}): {stats['filtered_boxes']}")

    if stats['total_images'] > 0:
        avg_boxes = stats['total_boxes'] / stats['total_images']
        print(f"\n📈 Average boxes per image: {avg_boxes:.1f}")

    print(f"\n✅ Annotated images saved to: {OUTPUT_DIR}/")
    print(f"   Total files: {len(list(OUTPUT_DIR.glob('*.jpg')))}")

    print("\n💡 Tips:")
    print(f"   - Open {OUTPUT_DIR}/ to view annotated images")
    print(f"   - Green = Carpetweed, Blue = Morning Glory, Red = Palmer Amaranth")
    print(f"   - Each box shows class name and confidence score")
    print(f"   - Summary stats shown at top of each image")

    if stats['low_conf_boxes'] > stats['high_conf_boxes']:
        print(f"\n⚠️  WARNING: Many low-confidence predictions detected!")
        print(f"   Consider setting CONFIDENCE_THRESHOLD = 0.25 in predict.py")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    visualize_predictions()
