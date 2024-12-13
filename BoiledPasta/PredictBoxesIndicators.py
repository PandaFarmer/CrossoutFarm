import numpy as np
import cv2
from ColorCount import IndicatorProperties, pixel_count_over_threshold, color_count
from Utilities import tile_split

def test_pixel_ratios():
    for filepath, val in IndicatorProperties.items():
        # Image.open()
        im = cv2.imread("Testing/CroppedIndicators/" + filepath)
        print(f"filepath: {filepath}")
        print(f"val: {val}")
        Margin = .05
        subdivided_pixel_ratios = subdivided_pixel_ratios(im, val["dominant_color"], val["div_dims"])
        print(f"subdivided_pixel_ratios: {subdivided_pixel_ratios.transpose()}")
        subdivided_pixel_ratios_compare = subdivided_pixel_ratios(im, val["dominant_color"], val["div_dims"])
        result, confidence = pixel_ratio_match_result(subdivided_pixel_ratios_compare, subdivided_pixel_ratios, Margin)
        print(f"pixel ratio match result: {result} confidence: {confidence}")
        # for filepathCompare, valCompare in IndicatorProperties.items():
        #     imCompare = cv2.imread("Testing/CroppedIndicators/" + filepathCompare)
        #     subdivided_pixel_ratios_compare = subdivided_pixel_ratios(im, valCompare["dominant_color"], valCompare["div_dims"])
        #     print(f"comparing: {filepath}, {filepathCompare}")
        #     result, confidence = pixel_ratio_match_result(subdivided_pixel_ratios_compare, subdivided_pixel_ratios, Margin)
        

#few number of classes, easily defined features with extremely low variation + no trust in premade being able to run
#low overhead solution for this specific function
def model_process_indicators(input_tensor):

    div_dims = []
    image_shapes = []
    image_axis_dims = []
    scale_start, scale_end = .5, 2
    
    subdivided_pixel_ratios_dict = dict()
    
    
    #preprocess and store Indicator Image Properties
    for filepath, val in IndicatorProperties.items():
        im = cv2.imread("Testing/CroppedIndicators/" + filepath)
        subdivided_pixel_ratios = subdivided_pixel_ratios(im, val["dominant_color"], val["div_dims"])
        subdivided_pixel_ratios_dict[filepath] = subdivided_pixel_ratios
        image_shapes.append(im.shape)
        image_axis_dims.extend(im.shape)
    
    x_max, y_max = input_tensor.shape[0], input_tensor.shape[1]
    max_image_dim = max(image_axis_dims)
    grid_anchor_dim = min(input_tensor.shape)//3

    anchor_search_queue = [] #format is tuple of (x, y, Height, Width)
    
    #uncomment if you need offsets during search
    # base_offset = grid_anchor_dim*.5
    # offsets = [(0, 0), (base_offset, 0), (0, base_offset), (base_offset, base_offset)]
    # for offsetX, offsetY in offsets:
    #     for x in range(offsetX, x_max, grid_anchor_dim):
    #         for y in range(offsetY, y_max, grid_anchor_dim):
    #             anchor_search_queue.append(())
    
    # def apply_across_grid_xy(fn, offsetX=0, offsetY=0):
    #     for x in range(offsetX, x_max, grid_anchor_dim):
    #         for y in range(offsetY, y_max, grid_anchor_dim):
    #             fn(x, y) 
    # add_anchor_to_search_q = lambda x, y: anchor_search_queue.append((x, y, grid_anchor_dim, grid_anchor_dim))
    # apply_across_grid_xy(add_anchor_to_search_q)
    
    
    
    predict_boxes = []
    
    def search_for_matching_indicator(indicator_filepath):
        indicator_val = IndicatorProperties[indicator_filepath]
        
        #consider changing grid_anchor_dims to fit indicator_val dim_divs?
        for x in range(0, x_max, grid_anchor_dim):
            for y in range(0, y_max, grid_anchor_dim):
                anchor_search_queue.append(((x, y), (x+grid_anchor_dim, y+grid_anchor_dim)))
                
        while anchor_search_queue:
            _x, _y, _h, _w = anchor_search_queue.pop(0)
            
            # add_anchor_to_search_q = lambda x, y: anchor_search_queue.append((x, y, _h, grid_anchor_dim))
            # start = grid_anchor_dim*offset
            cropped_image = im[_x:_x+_w, _y:_y+_h]

            if pixel_count_over_threshold(cropped_image, indicator_val["dominant_color"], indicator_val["_count"]):
                if min(cropped_image.shape) < max_image_dim*2:
                    predict_boxes.append(cropped_image, (_x, _y, _h, _w), indicator_filepath)
                else:
                    half_h, half_w = _h//2, _w//2
                    anchor_search_queue.append(x, y, half_h, half_w)
                    anchor_search_queue.append(x+half_w, y, half_h, half_w)
                    anchor_search_queue.append(x, y+half_h, half_h, half_w)
                    anchor_search_queue.append(x+half_w, y+half_h, half_h, half_w)
    
    for FilePath, val in IndicatorProperties.items():
        search_for_matching_indicator(FilePath)           
    
    #TODO process predict_boxes with atomic anchors
        
    return predict_boxes

# def process_atomic_anchor():
#     def process(x_offset, y_offset, ):
#         for x in range(x_max):
#             for y in range(y_max):
#                 if x+x_offset >= x_max or y+y_offset >= y_max:
#                     return
        
        
#     for scale in range(scale_start, scale_end, .1):
#         for x_offset in range(grid_anchor_dim, 3):
#             for y_offset in range(grid_anchor_dim, 3):
#                 process(x_offset, y_offset, )

#returns a 2d ndarrays of pixel count ratios per subdivision/tile
def subdivided_pixel_ratios(im, color, divs):
    # assert(im.shape[0]%xdiv == 0)
    # assert(im.shape[1]%ydiv == 0)
    upsample = 2
    ydiv, xdiv = divs
    
    ydiv *=upsample
    xdiv *=upsample
    
    M = im.shape[0]//xdiv
    N = im.shape[1]//ydiv
    
    tiles = [[im[x:x+M,y:y+N] for x in range(0,im.shape[0],M)] for y in range(0,im.shape[1],N)]
    
    #shaving off thin edge tiles
    _m, _n = M*.3, N*.3
    tiles = [[tile for tile in tile_row if tile.shape[0] >= _m and tile.shape[1] >= _n] for tile_row in tiles]
    tiles = [tile_row for tile_row in tiles if tile_row]
    
    pixel_counts = [[color_count(tile, color) for tile in tile_row] for tile_row in tiles]
    pixel_counts = np.asarray(pixel_counts, dtype=np.float32)
    pixel_ratios = 1/(M*N) * pixel_counts
    
    import matplotlib.pyplot as plt
    
    # for y, tile_row in enumerate(tiles):
    #     for x, tile in enumerate(tile_row):
    #         print(f"x, y: {x}, {y}")
    #         plt.imshow(tile)
    #         plt.show()
    #         plt.cla()
    
    return pixel_ratios #div to get normalized ratios
    
def pixel_ratio_match_result(PixelRatioArrCompare, PixelRatioArrOriginal, Margin):
    assert(PixelRatioArrCompare.shape == PixelRatioArrOriginal.shape)
    diff = PixelRatioArrOriginal - PixelRatioArrCompare
    margin_arr = np.full(
            shape=diff.shape,
            fill_value=Margin,
            dtype=np.float32
            )
    isMatching = not np.any(diff > margin_arr)
    fn = np.vectorize(lambda x: max(-x, 0))

    diff = fn(diff)
    confidence = 1 - np.sum(diff)/diff.size
    return isMatching, confidence
    
#compares 2nparrays describing pixel ratios per subdivision/tile
#main issue cutoff of edge along gridlines.. +issues produced by varying scaling/offset
def pixel_ratios_within_margins(PixelRatioArr1, PixelRatioArr2, Margin):
    assert(PixelRatioArr1.shape == PixelRatioArr2.shape)
    diff = PixelRatioArr1 - PixelRatioArr2
    margin_arr = np.full(
            shape=diff.shape,
            fill_value=Margin,
            dtype=np.float32
            )
    return not np.any(diff > margin_arr)

def match_indicator(pathToIndicatorImage, newImage):
    
    # im = Image.open(pathToIndicatorImage)
    im = cv2.imread(pathToIndicatorImage)
    
    M = im.shape[0]//2
    N = im.shape[1]//2
    
    IndicatorVal = IndicatorProperties[pathToIndicatorImage]
    IndicatorColor = IndicatorVal["dominant_color"]
    IndicatorThresholdCount = IndicatorVal["_count"]
    IndicatorThresholdRatio = _count/im.size
    
    tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]

    pixel_count_over_threshold(im, IndicatorColor, IndicatorThresholdCount*.97)

    
    
#BlueAlliedTriangle.png, dominant_color: [ 69 163 255], _count: 44
#EscortMissionIndicator.png, dominant_color: [255 150 0], _count: 630
#PurpleCrosshair.png, dominant_color: [191 114 252], _count: 48
#RedChevron.png, dominant_color: [255  50  50], _count: 80
    
    
def predict_boxes(self, predict_boxes: torch.Tensor) -> np.ndarray:
    """Convert network outputs to bounding boxes"""
    batch_boxes = []
    batch_scores = []
    batch_classes = []
    
    # Process each item in the batch
    for batch_idx in range(predict_boxes.shape[0]):
        boxes = []
        scores = []
        class_ids = []
        
        # Process each feature map location
        feature_map = predict_boxes[batch_idx]
        
        # Get grid size
        grid_size = feature_map.shape[1:3]
        
        # For each anchor in the grid
        for y in range(grid_size[0]):
            for x in range(grid_size[1]):
                # For each anchor
                for anchor_idx, anchor in enumerate(self.anchors):
                    # Get predictions for this anchor/anchor combination
                    pred = feature_map[anchor_idx * (5 + len(self.classes)):
                                    (anchor_idx + 1) * (5 + len(self.classes)),
                                    y, x]
                    
                    # Object confidence
                    obj_conf = torch.sigmoid(pred[4])
                    
                    # Class predictions
                    class_pred = torch.softmax(pred[5:], dim=0)
                    class_score, class_id = torch.max(class_pred, dim=0)
                    
                    # Combined score
                    score = float(obj_conf * class_score)
                    
                    if score > self.confidence_threshold:
                        # Convert network outputs to box coordinates
                        bx = (torch.sigmoid(pred[0]) + x) / grid_size[1]
                        by = (torch.sigmoid(pred[1]) + y) / grid_size[0]
                        bw = torch.exp(pred[2]) * anchor[2] / self.input_size[1]
                        bh = torch.exp(pred[3]) * anchor[3] / self.input_size[0]
                        
                        # Convert to corner format
                        x1 = bx - bw/2
                        y1 = by - bh/2
                        x2 = bx + bw/2
                        y2 = by + bh/2
                        
                        boxes.append([x1, y1, x2, y2])
                        scores.append(score)
                        class_ids.append(int(class_id))
        
        batch_boxes.append(boxes)
        batch_scores.append(scores)
        batch_classes.append(class_ids)
        
    return np.array(batch_boxes), np.array(batch_scores), np.array(batch_classes)
        
if __name__ == "__main__":
    test_pixel_ratios()