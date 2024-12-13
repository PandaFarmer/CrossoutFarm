import cv2
import numpy as np
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from PredictBoxesIndicators import model_process_indicators, predict_boxes

class ObjectDetector:
    def __init__(self, 
                 model_path: str,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 input_size: Tuple[int, int] = (416, 416)):
        """
        Initialize the object detector
        
        Args:
            model_path: Path to trained model weights
            confidence_threshold: Minimum confidence score for detection
            nms_threshold: IoU threshold for non-max suppression
            input_size: Model input dimensions (height, width)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
        # Load your model here
        self.model = self._load_model(model_path)
        
        # Define anchor boxes for different scales
        self.anchors = self._generate_anchors()
        
        # Load class names
        self.classes = self._load_classes()

    def _load_model(self, model_path: str) -> nn.Module:
        """Load the neural network model"""
        # This is a placeholder - implement according to your model architecture
        model = torch.load(model_path)
        model.eval()
        return model

    def _generate_anchors(self) -> np.ndarray:
        """Generate anchor boxes for different scales"""
        scales = [1, 2, 4]
        ratios = [0.5, 1.0, 2.0]
        base_size = 16
        anchors = []
        
        for scale in scales:
            for ratio in ratios:
                h = base_size * scale * np.sqrt(ratio)
                w = base_size * scale * np.sqrt(1.0/ratio)
                anchors.append([-w/2, -h/2, w/2, h/2])
                
        return np.array(anchors)

    def _load_classes(self) -> List[str]:
        """Load class names"""
        # Implement based on your classes file
        return ['person', 'car', 'dog']  # example classes

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Resize image
        resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = rgb / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.permute(2, 0, 1)  # HWC to CHW format
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor


    def non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray, 
                          class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply non-max suppression to remove overlapping boxes"""
        final_boxes = []
        final_scores = []
        final_classes = []
        
        # Process each class separately
        unique_classes = np.unique(class_ids)
        
        for cls in unique_classes:
            # Get boxes for this class
            class_mask = class_ids == cls
            cls_boxes = boxes[class_mask]
            cls_scores = scores[class_mask]
            
            # Sort by confidence
            indices = np.argsort(cls_scores)[::-1]
            
            kept_indices = []
            while len(indices) > 0:
                # Keep highest scoring box
                kept_indices.append(indices[0])
                
                # Calculate IoU with remaining boxes
                ious = np.array([self.calculate_iou(cls_boxes[indices[0]], 
                                                  cls_boxes[idx]) 
                               for idx in indices[1:]])
                
                # Remove boxes with high IoU
                indices = indices[1:][ious <= self.nms_threshold]
            
            final_boxes.extend(cls_boxes[kept_indices])
            final_scores.extend(cls_scores[kept_indices])
            final_classes.extend([cls] * len(kept_indices))
        
        return np.array(final_boxes), np.array(final_scores), np.array(final_classes)

    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection)

    def postprocess_boxes(self, boxes: np.ndarray, 
                         original_shape: Tuple[int, int]) -> np.ndarray:
        """Scale boxes back to original image size"""
        height_scale = original_shape[0] / self.input_size[0]
        width_scale = original_shape[1] / self.input_size[1]
        
        scaled_boxes = boxes.copy()
        scaled_boxes[:, [0, 2]] *= width_scale
        scaled_boxes[:, [1, 3]] *= height_scale
        
        return scaled_boxes

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in an image
        
        Args:
            image: Input image in BGR format (OpenCV)
            
        Returns:
            List of dictionaries containing detection results:
            [{'box': [x1, y1, x2, y2], 'score': confidence, 'class': class_name}, ...]
        """
        # Save original image shape
        original_shape = image.shape[:2]
        
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        # with torch.no_grad():
        #     feature_maps = self.model(input_tensor)
        
        # # Get boxes, scores, and class ids
        # boxes, scores, class_ids = self.predict_boxes(feature_maps)
        
        boxes, scores, class_ids = model_process_indicators(input_tensor)
        
        # Apply NMS
        boxes, scores, class_ids = self.non_max_suppression(boxes, scores, class_ids)
        
        # Scale boxes back to original image size
        boxes = self.postprocess_boxes(boxes, original_shape)
        
        # Prepare results
        results = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            results.append({
                'box': box.tolist(),
                'score': float(score),
                'class': self.classes[class_id]
            })
            
        return results

# Usage example:
if __name__ == "__main__":
    # Initialize detector
    detector = ObjectDetector(
        model_path='model.pth',
        confidence_threshold=0.5,
        nms_threshold=0.4
    )
    
    # Load and process image
    image = cv2.imread('path/to/your/image.jpg')
    detections = detector.detect(image)
    
    # Draw results
    for det in detections:
        box = det['box']
        cv2.rectangle(image, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     (0, 255, 0), 2)
        
        label = f"{det['class']}: {det['score']:.2f}"
        cv2.putText(image, label, 
                   (int(box[0]), int(box[1] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show result
    cv2.imshow('Detections', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()