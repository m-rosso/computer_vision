############################################################################################################################################
# CROP:

# Original dimensions:
height, width, channels = original_img.shape

# Cropped dimensions:
cropped_height = height//2
cropped_width = width//2

# Cropped image:
cropped_img = original_img[cropped_height-100: cropped_height+100, cropped_width-100: cropped_width+100, :]

# Resized dimensions:
resized_width = 512
resized_height = 384

# Resized image:
resized_cropped_img = cv2.resize(
    cropped_img,
    (resized_width, resized_height),
    interpolation=cv2.INTER_AREA
)

############################################################################################################################################
# FLIP:

# Horizontal flipped image:
hor_flipped_img = cv2.flip(original_img, 1)

# Vertical flipped image:
ver_flipped_img = cv2.flip(original_img, 0)

# Horizontal and vertical flipped image:
hor_ver_flipped_img = cv2.flip(original_img, -1)

############################################################################################################################################
# ROTATION:

# Original dimensions:
height, width, channels = original_img.shape
center = (height//2, width//2)

# Rotation matrix:
M = cv2.getRotationMatrix2D(
    center,
    angle=45,
    # angle=90,
    scale=1.0
)

# Rotated image:
rotated_img = cv2.warpAffine(
    original_img,
    M,
    (width, height)
)

############################################################################################################################################
# BLUR:

blurred_img = cv2.blur(original_img, (9,9))

median_blurred_img = cv2.medianBlur(original_img, 11)

############################################################################################################################################
# MUDANÇA DE BRILHO:

bright_img = brightness_filter(image=original_img, delta=120)

bright_img = brightness_filter(image=original_img, delta=-120)

############################################################################################################################################
# MUDANÇA DE CONTRASTE:

contrast_img = contrast_filter(image=original_img, beta=120)

contrast_img = contrast_filter(image=original_img, beta=-120)

############################################################################################################################################
# MUDANÇA DE SATURAÇÃO:

saturation_img = saturation_filter(image=original_img, beta=120)

saturation_img = saturation_filter(image=original_img, beta=-120)
