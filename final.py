import cv2
import numpy as np
import argparse
import sys
import os
from datetime import datetime

def empty(p):
    pass

def cal_area(pixelArea, distance):
    normalized_pixel_area = pixelArea * pow(distance, 2)
    area_in_cm2 = normalized_pixel_area * 0.005318513
    return area_in_cm2


parser = argparse.ArgumentParser(description='Shape detection on an image.')
parser.add_argument('imagePath', type=str, help='Path to the image file')
parser.add_argument('--distance', type=int, default=1, help='Distance between camera and the shape')
parser.add_argument('--clusters', type=int, default=4, help='Number of clusters for color quantization')
args = parser.parse_args()

img = cv2.imread(args.imagePath)
if img is None:
    print("Error: Image not found.")
    sys.exit()

window_name = "Result"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, img.shape[1], img.shape[0])

cv2.namedWindow("Parameters", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 174, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 45, 255, empty)
cv2.createTrackbar("Area", "Parameters", 5000, 100000, empty)

def color_quantization(image, k):
    data = np.float32(image).reshape((-1, 3))
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    
    quantized_image = centers[labels.flatten()]
    quantized_image = quantized_image.reshape(image.shape)
    
    return quantized_image

def detectColor(img, contour):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    mean_color = cv2.mean(imgContour, mask=mask)
    color_text = f"Color: ({int(mean_color[2])}, {int(mean_color[1])}, {int(mean_color[0])})"
    return color_text

def getContours(img, imgContour):
    newimg = cv2.resize(img, (1024, 1024))
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    conts, _ = cv2.findContours(newimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cnt, cn in zip(contours, conts):
        area = cv2.contourArea(cn)
        area_in_cm2 = cal_area(area, args.distance)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 4)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            
            color_text = detectColor(imgContour, cnt)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area_in_cm2)) + ' cm ^ 2', (x, y + h + 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(imgContour, color_text, (x, y + h // 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

quantized = False
quant_img = color_quantization(img.copy(), args.clusters)
while True:
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    
    imgContour = img.copy() if not quantized else quant_img.copy()

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if not quantized else cv2.cvtColor(quant_img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(imgHSV, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(imgHSV, lower_red, upper_red)

    red_mask = mask1 + mask2

    imgRedMasked = cv2.bitwise_and(img, img, mask=red_mask) if not quantized else cv2.bitwise_and(quant_img, quant_img, mask=red_mask)

    imgBlurRed = cv2.GaussianBlur(imgRedMasked, (7, 7), 1)
    imgGrayRed = cv2.cvtColor(imgBlurRed, cv2.COLOR_BGR2GRAY)
    imgCannyRed = cv2.Canny(imgGrayRed, threshold1=threshold1, threshold2=threshold2)
    kernel = np.ones((5, 5))
    imgDilRed = cv2.dilate(imgCannyRed, kernel=kernel, iterations=2)

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1) if not quantized else cv2.GaussianBlur(quant_img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, threshold1=threshold1, threshold2=threshold2)
    imgDil = cv2.dilate(imgCanny, kernel=kernel, iterations=1)

    getContours(imgDilRed, imgContour)
    getContours(imgDil, imgContour)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%H:%M:%S")
    date = current_time.strftime("%Y-%m-%d")
    height, width = imgContour.shape[:2]

    cv2.putText(imgContour, "Distance: " + str(int(args.distance)) + ' m', (10, height - 90), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(imgContour, "Date: " + str(date), (10, height - 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow(window_name, imgContour)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('s'):
        output_image = imgContour.copy()
        image_path = args.imagePath
        print(image_path)
        base_name = os.path.basename(image_path)
        raw_name, ext = os.path.splitext(base_name)
        output_image_name = "./outputImages/" + raw_name + '.output' + ext
        
        cv2.imwrite(output_image_name, output_image)
        print(f"Image saved as {output_image_name}.")
    elif key == ord('c'):
        if quantized:
            quantized = False
        else:
            quantized = True
    elif key == ord('q'):
        break


cv2.destroyAllWindows()
