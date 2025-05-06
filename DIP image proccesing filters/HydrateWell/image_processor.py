import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
from scipy import signal

def convert_to_grayscale(image):
    """
    Convert a color image to grayscale using PIL.
    
    Args:
        image (PIL.Image): The original color image
        
    Returns:
        PIL.Image: The grayscale version of the image
    """
    # Make a copy of the image to avoid modifying the original
    # Then convert to grayscale using PIL's built-in method
    grayscale_image = image.copy().convert('L')
    return grayscale_image

def convert_to_grayscale_weighted(image):
    """
    Convert a color image to grayscale using weighted RGB channels.
    This method gives more control over the conversion process.
    
    Args:
        image (PIL.Image): The original color image
        
    Returns:
        PIL.Image: The grayscale version of the image
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Check if the image is already grayscale
    if len(img_array.shape) < 3:
        return image
    
    # Apply weighted formula: Y = 0.299*R + 0.587*G + 0.114*B
    # This formula is based on human perception of color
    if img_array.shape[2] >= 3:  # Make sure it has RGB channels
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        grayscale_array = gray.astype(np.uint8)
        
        # Convert numpy array back to PIL image
        grayscale_image = Image.fromarray(grayscale_array)
        return grayscale_image
    
    # Fallback to PIL's convert method if shape is unexpected
    return image.convert('L')

def apply_gaussian_blur(image, sigma=1.0):
    """
    Apply Gaussian blur filter to the image.
    
    The Gaussian blur uses a 2D Gaussian function:
    G(x,y) = (1/(2*pi*sigma^2)) * e^(-(x^2+y^2)/(2*sigma^2))
    
    Args:
        image (PIL.Image): The original image
        sigma (float): Standard deviation of the Gaussian distribution
        
    Returns:
        PIL.Image: Blurred image
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Apply Gaussian filter
    blurred_array = ndimage.gaussian_filter(img_array, sigma=(sigma, sigma, 0) if len(img_array.shape) == 3 else sigma)
    
    # Convert back to PIL image
    blurred_image = Image.fromarray(blurred_array.astype(np.uint8))
    return blurred_image

def apply_sobel_edge_detection(image):
    """
    Apply Sobel edge detection to the image.
    
    Sobel operator uses two 3x3 kernels to compute gradient approximations:
    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    Then gradient magnitude = sqrt(Gx^2 + Gy^2)
    
    Args:
        image (PIL.Image): The original image (will be converted to grayscale)
        
    Returns:
        PIL.Image: Edge detected image
    """
    # Convert to grayscale first
    grayscale = image.convert('L')
    img_array = np.array(grayscale)
    
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Apply convolution
    grad_x = signal.convolve2d(img_array, sobel_x, mode='same', boundary='symm')
    grad_y = signal.convolve2d(img_array, sobel_y, mode='same', boundary='symm')
    
    # Calculate gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-255
    grad_magnitude = 255 * grad_magnitude / np.max(grad_magnitude)
    
    # Convert back to PIL image
    edge_image = Image.fromarray(grad_magnitude.astype(np.uint8))
    return edge_image

def apply_laplacian_edge_detection(image):
    """
    Apply Laplacian edge detection to the image.
    
    Laplacian kernel for edge detection:
    Laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    
    Args:
        image (PIL.Image): The original image (will be converted to grayscale)
        
    Returns:
        PIL.Image: Edge detected image
    """
    # Convert to grayscale first
    grayscale = image.convert('L')
    img_array = np.array(grayscale)
    
    # Define Laplacian kernel
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
    # Apply convolution
    edge_array = signal.convolve2d(img_array, laplacian, mode='same', boundary='symm')
    
    # Take absolute values and normalize to 0-255
    edge_array = np.abs(edge_array)
    edge_array = 255 * edge_array / np.max(edge_array)
    
    # Convert back to PIL image
    edge_image = Image.fromarray(edge_array.astype(np.uint8))
    return edge_image

def apply_sharpening(image, alpha=1.5):
    """
    Apply sharpening filter to the image.
    
    Sharpening uses the formula: output = original + alpha * (original - blurred)
    
    Args:
        image (PIL.Image): The original image
        alpha (float): Strength of the sharpening effect
        
    Returns:
        PIL.Image: Sharpened image
    """
    # Convert to numpy array
    img_array = np.array(image).astype(float)
    
    # Create blurred version
    blurred_array = ndimage.gaussian_filter(img_array, sigma=(1, 1, 0) if len(img_array.shape) == 3 else 1)
    
    # Calculate sharpened image
    sharpened = img_array + alpha * (img_array - blurred_array)
    
    # Clip values to valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Convert back to PIL image
    sharpened_image = Image.fromarray(sharpened)
    return sharpened_image

def apply_histogram_equalization(image):
    """
    Apply histogram equalization to enhance contrast.
    
    The transformation function is: T(r) = (L-1) * CDF(r)
    where L is the number of gray levels and CDF is the cumulative distribution function.
    
    Args:
        image (PIL.Image): The original image (will be converted to grayscale)
        
    Returns:
        PIL.Image: Contrast enhanced image
    """
    # Convert to grayscale
    grayscale = image.convert('L')
    img_array = np.array(grayscale)
    
    # Calculate histogram and CDF
    hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    
    # Normalize CDF
    cdf_normalized = cdf * 255 / cdf[-1]
    
    # Create lookup table
    lut = np.interp(np.arange(256), bins[:-1], cdf_normalized)
    
    # Apply equalization
    equalized_array = lut[img_array].astype(np.uint8)
    
    # Convert back to PIL image
    equalized_image = Image.fromarray(equalized_array)
    return equalized_image

def apply_binary_threshold(image, threshold=127):
    """
    Apply binary thresholding to the image.
    
    Binary thresholding formula:
    f(x,y) = 255 if pixel(x,y) > threshold, else 0
    
    Args:
        image (PIL.Image): The original image (will be converted to grayscale)
        threshold (int): Threshold value (0-255)
        
    Returns:
        PIL.Image: Binary image
    """
    # Convert to grayscale
    grayscale = image.convert('L')
    img_array = np.array(grayscale)
    
    # Apply threshold
    binary_array = (img_array > threshold) * 255
    
    # Convert back to PIL image
    binary_image = Image.fromarray(binary_array.astype(np.uint8))
    return binary_image

def apply_median_filter(image, kernel_size=3):
    """
    Apply median filter to remove noise while preserving edges.
    
    The median filter replaces each pixel with the median value in its neighborhood.
    
    Args:
        image (PIL.Image): The original image
        kernel_size (int): Size of the neighborhood window
        
    Returns:
        PIL.Image: Filtered image
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Apply median filter
    if len(img_array.shape) == 3:  # Color image
        filtered_array = np.zeros_like(img_array)
        for i in range(img_array.shape[2]):  # Process each channel
            filtered_array[:, :, i] = ndimage.median_filter(img_array[:, :, i], size=kernel_size)
    else:  # Grayscale image
        filtered_array = ndimage.median_filter(img_array, size=kernel_size)
    
    # Convert back to PIL image
    filtered_image = Image.fromarray(filtered_array.astype(np.uint8))
    return filtered_image
