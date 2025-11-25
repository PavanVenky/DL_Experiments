import os
import cv2
import pandas as pd

def drawbounding_boxes():
    image_dir = r"D:\Pavan\My_Python_Project\Tiny_LISA"
    csv_path = r"D:\Pavan\My_Python_Project\Tiny_LISA\annotations.csv"
    output_dir = os.path.join(image_dir, "output")

    # Create output folder if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Load annotations
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        filename = row['filename']
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        cls = row['class']

        img_path = os.path.join(image_dir, filename)

        if not os.path.isfile(img_path):
            print(f"Image not found: {img_path}")
            continue

        # Load image
        img = cv2.imread(img_path)

        # Draw bounding box (green rectangle)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put class label above box
        cv2.putText(img, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Save image
        out_path = os.path.join(output_dir, f"boxed_{filename}")
        cv2.imwrite(out_path, img)

        print(f"Saved: {out_path}")

    print("DONE — All bounding boxes drawn.")

drawbounding_boxes()