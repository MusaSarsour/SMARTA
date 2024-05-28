import sys
import cv2
from mtcnn import MTCNN
import time
import facealignment

# Define a function to resize the image while keeping the aspect ratio
def resize_image(image, desired_width, desired_height):
    """Resize an image while maintaining aspect ratio."""
    aspect_ratio = image.shape[1] / image.shape[0]
    if aspect_ratio > 1:  # Wider than tall
        new_width = desired_width
        new_height = int(desired_width / aspect_ratio)
    else:  # Taller than wide
        new_height = desired_height
        new_width = int(desired_height * aspect_ratio)
    return cv2.resize(image, (new_width, new_height))

# Main function to process and save the detected faces
def process_image(image_path, output_folder='comp_img', desired_width=224, desired_height=224):
    # Load the image
    img = cv2.imread(image_path)

    # Initialize the face detector (MTCNN)
    detector = MTCNN()

    # Detect faces
    start = time.time()
    detections = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Set minimum confidence level
    min_conf = 0.9

    # Initialize the face alignment tool (make sure the class exists and is properly initialized)
    tool = facealignment.FaceAlignmentTools(steps_threshold=[0.1, 0.1, 0.1], min_face_size=5)

    # Process each detection
    for i, det in enumerate(detections):
        if det['confidence'] >= min_conf:
            x, y, width, height = det['box']

            # Crop the detected face with a margin
            face_img = img[max(0, y-3):y+height+3, max(0, x-3):x+width+3]

            # Align the face (make sure the alignment tool can accept cv2 format)
            aligned_face = tool.align(face_img)

            # Resize the aligned face
            aligned_face_resized = resize_image(aligned_face, desired_width, desired_height)

            # Convert BGR to RGB if needed by your tool or for visualization
            aligned_face_rgb = cv2.cvtColor(aligned_face_resized, cv2.COLOR_BGR2RGB)

            # Save the aligned face with resizing
            face_img_path = f'{output_folder}/Test_{i + 66:03d}.jpg'
            cv2.imwrite(face_img_path, cv2.cvtColor(aligned_face_rgb, cv2.COLOR_RGB2BGR))

    # Calculate and print the processing time
    end = time.time()
    print(f"Detected {len(detections)} faces in {end - start} seconds.")

if __name__ == '__main__':
    image_path = sys.argv[1]  # Input image path
    process_image(image_path)
