import cv2
import numpy as np

def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 灰階
    
    img = cv2.equalizeHist(img) # 直方圖均衡化 - 增強對比度和細節
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # 自適應直方圖均衡化-處理暗部和亮部細節
    img = clahe.apply(img)
    
    img = cv2.medianBlur(img, 3) # 中值濾波(椒鹽雜訊)

    img = cv2.GaussianBlur(img, (5, 5), 0) # 高斯模糊(高頻雜訊)
        
    img = cv2.bilateralFilter(img, 3, 5, 5) # 雙邊濾波(保留邊緣)
    
    gamma = 0.5  # 改善暗部細節
    img = np.array(255 * (img / 255) ** gamma, dtype=np.uint8) # gamma校正

    return img

def detect_edges(image, method='sobel'):
    if method.lower() == 'sobel':
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

        edge = np.sqrt(sobel_x**2 + sobel_y**2)
        
        edge = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        kernel = np.ones((3,3), np.uint8) # 簡化線條
        
        # edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel) # 閉運算，填補小的斷裂
        
        # edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, kernel)  # 開運算，去除細小雜點
        
    elif method.lower() == 'canny':
        edge = cv2.Canny(image, 100, 200)
        
    return edge

def apply_thresholding(edges, threshold_type='adaptive'): 
    if threshold_type.lower() == 'binary':
        _, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY_INV)
        
    elif threshold_type.lower() == 'adaptive':
        binary_edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
  
    
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