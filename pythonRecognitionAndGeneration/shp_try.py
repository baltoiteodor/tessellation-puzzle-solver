import cv2
import numpy as np

def detect_shapes(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to get a binary image
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.Canny(gray, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_shapes = []
    
    # Filter contours based on area and draw bounding rectangles
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 10000:  # Adjust the area range based on your expected shape sizes
            # Get minimum bounding rectangle to straighten the shape
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Draw minimum bounding rectangle on the original image
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
            
            # Get shape properties
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add shape details to the list
            detected_shapes.append({
                "Bounding Box": (x, y, w, h),
                "Area": area,
                "Perimeter": cv2.arcLength(contour, True)
            })
    
    # Save the image with added bounding rectangles
    cv2.imwrite("output_image_with_rectangles.jpg", image)
    
    return detected_shapes

# Replace 'your_image_path.jpg' with the path to your image
shapes = detect_shapes('images/original_puzzle.jpg')

# Print detected shapes and their details
for idx, shape in enumerate(shapes):
    print(f"Shape {idx + 1}:")
    print(f"Bounding Box: {shape['Bounding Box']}")
    print(f"Area: {shape['Area']}")
    print(f"Perimeter: {shape['Perimeter']}")
    print("-------------------------")

# Display the image with added rectangles
cv2.imshow("Image with Bounding Rectangles", cv2.imread("output_image_with_rectangles.jpg"))
cv2.waitKey(0)
cv2.destroyAllWindows()
