import os
import cv2

def tile_image(image_path, tile_size=(640, 640), grid=(3, 3), overlap=30):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    tiles = []

    tile_w = w // grid[0]
    tile_h = h // grid[1]

    for i in range(grid[1]):
        for j in range(grid[0]):
            x = j * (tile_w - overlap)
            y = i * (tile_h - overlap)
            x = max(0, min(x, w - tile_w))
            y = max(0, min(y, h - tile_h))
            tile = image[y:y+tile_h, x:x+tile_w]
            tiles.append(tile)
    return tiles

def save_tiles(tiles, output_dir="tiles", base_name="tile"):
    os.makedirs(output_dir, exist_ok=True)
    for idx, tile in enumerate(tiles):
        path = os.path.join(output_dir, f"{base_name}_{idx}.jpg")
        cv2.imwrite(path, tile)
