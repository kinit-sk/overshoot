from PIL import Image, ImageDraw



img1 = "graph1_base_loss.png"
img2 = "graph1_test_loss.png"

def stitch_images_vertically_with_line(image_path1, image_path2, output_path, dash_length=10, space_length=10, line_thickness=1, padding=20):
    # Load the images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Determine the size for the new image, including space for the black line and padding
    width = max(image1.width, image2.width)
    height = image1.height + image2.height + line_thickness + padding

    # Create a new image with the appropriate height
    new_image = Image.new('RGB', (width, height), 'white')

    # Paste image1 at the top with padding
    # new_image.paste(image1, (0, padding))
    new_image.paste(image1, (0, 0))

    # Create a draw object
    draw = ImageDraw.Draw(new_image)

    # # Draw the black line below the first image with padding
    # line_y_position = image1.height + padding
    # draw.line([(0, line_y_position), (width, line_y_position)], fill="black", width=line_thickness)
    
    
    # Draw a dashed line below the first image with padding
    line_y_position = image1.height + padding
    x_start = 0
    while x_start < width:
        x_end = min(x_start + dash_length, width)
        draw.line([(x_start, line_y_position), (x_end, line_y_position)], fill="black", width=line_thickness)
        x_start += dash_length + space_length
    

    # Paste image2 below image1, after the black line and padding
    image2_y_position = image1.height + line_thickness +  padding
    new_image.paste(image2, (0, image2_y_position))

    # Save the new image
    new_image.save(output_path)

# Example usage
stitch_images_vertically_with_line(img1, img2, "graph1_join.png")