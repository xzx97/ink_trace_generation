from PIL import Image, ImageDraw

def draw_piano(filename):
    width, height = 800, 400
    # Create a white background image
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    
    # Draw white keys
    num_white_keys = 14
    key_width = width // num_white_keys
    for i in range(num_white_keys):
        x0 = i * key_width
        y0 = 0
        x1 = x0 + key_width
        y1 = height
        draw.rectangle([x0, y0, x1, y1], outline="black", width=2)
        
    # Draw black keys
    # Pattern of black keys: group of 2, group of 3
    black_key_positions = [1, 2, 4, 5, 6, 8, 9, 11, 12, 13]
    black_key_width = int(key_width * 0.6)
    black_key_height = int(height * 0.6)
    
    for pos in black_key_positions:
        x0 = pos * key_width - black_key_width // 2
        y0 = 0
        x1 = x0 + black_key_width
        y1 = black_key_height
        draw.rectangle([x0, y0, x1, y1], fill="black")

    # Save the image
    image.save(filename)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    draw_piano("test_gen_piano.jpg")
