import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 灰階
    
    median_filtered = cv2.medianBlur(gray, 3) # 中值濾波(椒鹽雜訊)

    blurred = cv2.GaussianBlur(median_filtered, (3, 3), 0) # 高斯模糊(高頻雜訊)
        
    # bilateral_filtered = cv2.bilateralFilter(blurred, 9, 75, 75) # 雙邊濾波(保留邊緣)

    return blurred

def detect_edges(image, method='sobel'):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return sobel_edges

def apply_thresholding(edges, threshold_type='binary'):
    _, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    
    return binary_edges

def create_image(image_path, output_path):
    
    image = cv2.imread(image_path)
    
    preprocessed = preprocess_image(image) # 預處理(灰階、高斯模糊)
    
    edges = detect_edges(preprocessed, method='sobel') # 偵測邊緣，使用sobal方法
    
    line_drawing = apply_thresholding(edges, threshold_type='binary') # 二值化處理
    
    cv2.imwrite(output_path, line_drawing)
    print(f"Please check {output_path}")

if __name__ == '__main__':
    input_path = [
        'input_image/yzu1.jpg', 
        'input_image/yzu2.jpg',
        'input_image/yzu3.jpg',
        'input_image/yzu4.jpg'
    ]
    
    output_path = [
        'output_image/output1.jpg', 
        'output_image/output2.jpg',
        'output_image/output3.jpg',
        'output_image/output4.jpg'
    ]
    
    for input_path, output_path in zip(input_path, output_path):
        create_image(input_path, output_path)