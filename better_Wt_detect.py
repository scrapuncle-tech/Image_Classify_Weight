import cv2
import pytesseract
import easyocr
import numpy as np
import re
import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import logging
from scipy import ndimage
from skimage import restoration, filters, exposure
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=True)  # Set to True if you have GPU

class DigitalDisplayDetector:
    """Specialized detector for digital displays with multiple color support"""
    
    @staticmethod
    def detect_colored_display_region(image, color_name, hsv_lower, hsv_upper):
        """Detect specific colored digital display regions"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            return (x, y, w, h)
        return None
    
    @staticmethod
    def detect_red_numbers(image):
        """Detect red numbers with improved contour grouping"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Filter contours
        valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 5000 and 
                          0.2 < (cv2.boundingRect(c)[2] / cv2.boundingRect(c)[3]) < 5]
        if not valid_contours:
            return None

        # Get bounding boxes and centers
        boxes = [cv2.boundingRect(c) for c in valid_contours]
        centers = [(x + w/2, y + h/2) for x, y, w, h in boxes]
        sorted_indices = sorted(range(len(centers)), key=lambda i: centers[i][0])
        sorted_contours = [valid_contours[i] for i in sorted_indices]
        sorted_boxes = [boxes[i] for i in sorted_indices]
        sorted_centers = [centers[i] for i in sorted_indices]

        # Calculate average dimensions
        avg_w = np.mean([w for _, _, w, _ in sorted_boxes])
        avg_h = np.mean([h for _, _, _, h in sorted_boxes])

        # Group contours
        groups = []
        current_group = [sorted_contours[0]]
        for i in range(1, len(sorted_contours)):
            dx = sorted_centers[i][0] - sorted_centers[i-1][0]
            dy = abs(sorted_centers[i][1] - sorted_centers[i-1][1])
            if dx < 1.5 * avg_w and dy < 0.5 * avg_h:
                current_group.append(sorted_contours[i])
            else:
                groups.append(current_group)
                current_group = [sorted_contours[i]]
        if current_group:
            groups.append(current_group)

        # Evaluate groups
        best_group = None
        max_contours = 0
        min_y_std = float('inf')
        for group in groups:
            if len(group) < 2:  # At least 2 contours for a number
                continue
            x_coords = [min(cv2.boundingRect(c)[0] for c in group)]
            y_coords = [min(cv2.boundingRect(c)[1] for c in group)]
            w_coords = [max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in group)]
            h_coords = [max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in group)]
            x, y = min(x_coords), min(y_coords)
            w, h = max(w_coords) - x, max(h_coords) - y
            aspect_ratio = w / h
            y_centers = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]/2 for c in group]
            y_std = np.std(y_centers) if len(y_centers) > 1 else 0

            if aspect_ratio > 2 and len(group) >= max_contours and y_std < min_y_std:
                best_group = (x, y, w, h)
                max_contours = len(group)
                min_y_std = y_std

        if best_group:
            x, y, w, h = best_group
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            return (x, y, w, h)
        
        # Fallback to largest contour
        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            return (x, y, w, h)
        return None
    
    @staticmethod
    def detect_bright_region(image):
        """Detect bright regions that might contain digital displays"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if area > 500 and 2 < aspect_ratio < 8:
                    valid_contours.append(contour)
            
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                return (x, y, w, h)
        return None

class BlurDetector:
    """Detect and measure blur in images"""
    
    @staticmethod
    def detect_blur(image):
        """Detect blur using multiple methods"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        gray = gray.astype(np.uint8)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2).mean()
        blur_score = {
            'laplacian_variance': laplacian_var,
            'gradient_magnitude': gradient_magnitude,
            'is_blurred': laplacian_var < 100 or gradient_magnitude < 10
        }
        return blur_score

class AdvancedDeblurrer:
    """Advanced deblurring techniques specifically for digital displays"""
    
    @staticmethod
    def safe_image_conversion(image):
        """Safely convert image to proper format"""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    return image.astype(np.uint8)
            else:
                return image.astype(np.uint8)
        return np.array(image, dtype=np.uint8)
    
    @staticmethod
    def enhance_for_ocr(image, region_name=None):
        """Enhance image for OCR based on display type"""
        image = AdvancedDeblurrer.safe_image_conversion(image)
        
        # Select appropriate channel
        if len(image.shape) == 3:
            if region_name and 'red' in region_name:
                channel = image[:,:,2]  # Red channel
            elif region_name and 'green' in region_name:
                channel = image[:,:,1]  # Green channel
            elif region_name and 'blue' in region_name:
                channel = image[:,:,0]  # Blue channel
            else:
                channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            channel = image

        enhanced_versions = []
        enhanced_versions.append(channel)  # Original channel

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_enhanced = clahe.apply(channel)
        enhanced_versions.append(clahe_enhanced)

        # Adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
        enhanced_versions.append(adaptive_thresh)

        # Unsharp masking
        gaussian = cv2.GaussianBlur(channel, (3,3), 0)
        unsharp = cv2.addWeighted(channel, 2.0, gaussian, -1.0, 0)  # Stronger sharpening
        unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
        enhanced_versions.append(unsharp)

        # Binary threshold for bright digits
        thresh = 0.8 * np.max(channel)
        _, binary = cv2.threshold(channel, thresh, 255, cv2.THRESH_BINARY)
        enhanced_versions.append(binary)

        # Inverted binary (for OCR flexibility)
        inverted_binary = cv2.bitwise_not(binary)
        enhanced_versions.append(inverted_binary)

        return enhanced_versions

def enhanced_tesseract_ocr(image):
    """Enhanced Tesseract OCR with configurations optimized for digital displays"""
    configs = [
        r'--oem 3 --psm 6',
        r'--oem 3 --psm 7',
        r'--oem 3 --psm 8',
        r'--oem 3 --psm 10',
        r'--oem 3 --psm 13',
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.',
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.',
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.',
    ]
    
    results = []
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(image, config=config).strip()
            if text:
                cleaned_text = re.sub(r'[^\d.]', '', text)
                if cleaned_text and len(cleaned_text) >= 1:
                    results.append(cleaned_text)
                numbers = re.findall(r'\d+\.?\d*', text)
                for num in numbers:
                    if len(num) >= 1:
                        results.append(num)
        except Exception as e:
            logger.debug(f"Tesseract config failed: {e}")
            continue
    
    return list(set(results))

def enhanced_easyocr_detection(image):
    """Enhanced EasyOCR with multiple confidence levels"""
    results = []
    try:
        for confidence_threshold in [0.05, 0.1, 0.2, 0.3, 0.5]:
            ocr_results = reader.readtext(image, detail=1, paragraph=False, 
                                        width_ths=0.5, height_ths=0.7)
            for bbox, text, conf in ocr_results:
                if conf > confidence_threshold:
                    results.append((text.strip(), conf, 'original'))
                    cleaned = re.sub(r'[^\d.]', '', text)
                    if cleaned and cleaned != text.strip():
                        results.append((cleaned, conf, 'cleaned'))
    except Exception as e:
        logger.error(f"EasyOCR failed: {e}")
    return results

def format_weight_smart(raw_text):
    """Smart weight formatting that preserves the actual reading"""
    if not raw_text:
        return None
    
    text = str(raw_text).strip().replace(' ', '').replace('O', '0').replace('o', '0')
    if re.match(r'^\d+\.?\d*$', text):
        try:
            num = float(text)
            if 0.001 <= num <= 999999:
                return text
        except:
            pass
    
    digits_only = re.sub(r'[^\d.]', '', text)
    if not digits_only or len(digits_only) < 1:
        return None
    
    if digits_only.count('.') > 1:
        parts = digits_only.split('.')
        digits_only = parts[0] + '.' + ''.join(parts[1:])
    
    try:
        num = float(digits_only)
        if 0.001 <= num <= 999999:
            return digits_only
    except:
        pass
    
    if '.' not in digits_only and len(digits_only) > 3:
        test_positions = [3, 2, 1]
        for pos in test_positions:
            if len(digits_only) > pos:
                formatted = digits_only[:-pos] + '.' + digits_only[-pos:]
                try:
                    num = float(formatted)
                    if 0.001 <= num <= 999999:
                        return formatted
                except:
                    continue
    
    try:
        num = float(digits_only)
        if 0.001 <= num <= 999999:
            return digits_only
    except:
        pass
    
    return None

def detect_weight_with_blur_handling(image_path):
    """Main detection function with improved digital display handling"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None
    
    display_regions = []
    green_region = DigitalDisplayDetector.detect_colored_display_region(
        image, "green", np.array([40, 50, 50]), np.array([80, 255, 255])
    )
    if green_region:
        display_regions.append(('green_display', green_region))
    
    blue_region = DigitalDisplayDetector.detect_colored_display_region(
        image, "blue", np.array([90, 50, 50]), np.array([130, 255, 255])
    )
    if blue_region:
        display_regions.append(('blue_display', blue_region))
    
    white_region = DigitalDisplayDetector.detect_colored_display_region(
        image, "white", np.array([0, 0, 200]), np.array([180, 30, 255])
    )
    if white_region:
        display_regions.append(('white_display', white_region))
    
    red_numbers = DigitalDisplayDetector.detect_red_numbers(image)
    if red_numbers:
        display_regions.append(('red_numbers', red_numbers))
    
    bright_region = DigitalDisplayDetector.detect_bright_region(image)
    if bright_region:
        display_regions.append(('bright_display', bright_region))
    
    if not display_regions:
        h, w = image.shape[:2]
        display_regions.append(('full_image', (0, 0, w, h)))
    
    blur_info = BlurDetector.detect_blur(image)
    
    results = {
        'image_path': image_path,
        'blur_info': blur_info,
        'detections': {},
        'best_detection': 'Not detected',
        'confidence': 0,
        'method': 'None',
        'regions_detected': len(display_regions)
    }
    
    all_detections = []
    
    for region_name, (x, y, w, h) in display_regions:
        roi = image[y:y+h, x:x+w]
        enhanced_versions = AdvancedDeblurrer.enhance_for_ocr(roi, region_name=region_name)
        
        for i, enhanced_img in enumerate(enhanced_versions):
            method_name = f"{region_name}_v{i+1}"
            tesseract_results = enhanced_tesseract_ocr(enhanced_img)
            easyocr_results = enhanced_easyocr_detection(enhanced_img)
            
            results['detections'][method_name] = {
                'tesseract': tesseract_results,
                'easyocr': [(text, conf) for text, conf, _ in easyocr_results]
            }
            
            for result in tesseract_results:
                formatted = format_weight_smart(result)
                if formatted:
                    conf_boost = 0.7 if 'display' in region_name else 0.5
                    if region_name == 'red_numbers':
                        conf_boost = 0.9
                    all_detections.append((formatted, conf_boost, f'Tesseract-{method_name}'))
            
            for text, conf, text_type in easyocr_results:
                formatted = format_weight_smart(text)
                if formatted:
                    if 'display' in region_name or 'red_numbers' in region_name:
                        conf = min(1.0, conf * 1.2)
                    all_detections.append((formatted, conf, f'EasyOCR-{method_name}-{text_type}'))
    
    if all_detections:
        all_detections.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        logger.info("Top detections:")
        for i, (text, conf, method) in enumerate(all_detections[:5]):
            logger.info(f"  {i+1}. {text} (conf: {conf:.2f}, method: {method})")
        
        best = all_detections[0]
        results['best_detection'] = best[0]
        results['confidence'] = best[1]
        results['method'] = best[2]
    
    return results

def save_results_to_csv(results, csv_path="enhanced_weight_results.csv"):
    """Save results to CSV with detailed information"""
    try:
        image_name = os.path.basename(results['image_path'])
        new_record = pd.DataFrame({
            "Image_Name": [image_name],
            "Detected_Weight": [results['best_detection']],
            "Confidence": [results['confidence']],
            "Detection_Method": [results['method']],
            "Is_Blurred": [results['blur_info']['is_blurred']],
            "Laplacian_Variance": [results['blur_info']['laplacian_variance']],
            "Gradient_Magnitude": [results['blur_info']['gradient_magnitude']],
            "Regions_Detected": [results['regions_detected']]
        })
        
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            updated_df = pd.concat([existing_df, new_record], ignore_index=True)
            updated_df = updated_df.drop_duplicates(subset=["Image_Name"], keep="last")
            updated_df.to_csv(csv_path, index=False)
        else:
            new_record.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def main():
    """Main function for single image detection"""
    logger.info("Starting enhanced digital display weight detection...")
    image_path = r"Copy of Image_3175.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n📷 Processing Image: {image_path}")
    results = detect_weight_with_blur_handling(image_path)
    
    best = results['best_detection']
    if best and isinstance(best, str) and '.' not in best and best.isdigit() and len(best) > 3:
        best = best[:-3] + '.' + best[-3:]
        results['best_detection'] = best
    
    if results:
        blur_status = "🌫️ BLURRED" if results['blur_info']['is_blurred'] else "✨ CLEAR"
        print(f"🔍 Image Quality: {blur_status}")
        print(f"📊 Blur Metrics:")
        print(f"   - Laplacian Variance: {results['blur_info']['laplacian_variance']:.2f}")
        print(f"   - Gradient Magnitude: {results['blur_info']['gradient_magnitude']:.2f}")
        print(f"🎯 Regions Found: {results['regions_detected']}")
        print(f"✅ Best Detection: {results['best_detection']}")
        print(f"🎯 Method: {results['method']}")
        print(f"📈 Confidence: {results['confidence']:.2f}")
        save_results_to_csv(results)
        print(f"💾 Results saved to enhanced_weight_results.csv")
    else:
        print(f"❌ Failed to process {image_path}")

if __name__ == "__main__":
    main()