import streamlit as st
import io
import numpy as np
from PIL import Image
from image_processor import (
    convert_to_grayscale, convert_to_grayscale_weighted, 
    apply_gaussian_blur, apply_sobel_edge_detection, 
    apply_laplacian_edge_detection, apply_sharpening,
    apply_histogram_equalization, apply_binary_threshold,
    apply_median_filter
)

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Digital Image Processing",
        layout="wide"
    )
    
    # Header
    st.title("Digital Image Processing")
    st.subheader("Apply Various Image Processing Filters")
    
    # Description
    st.markdown("""
    This application allows you to apply various digital image processing filters. 
    Simply upload an image, select a filter, adjust parameters if available, 
    and the app will show you both the original and processed versions side by side.
    """)
    
    # Sidebar for filter selection
    st.sidebar.title("Image Processing Options")
    filter_type = st.sidebar.selectbox(
        "Select Filter",
        [
            "Grayscale (Standard)",
            "Grayscale (Weighted)",
            "Gaussian Blur",
            "Sobel Edge Detection",
            "Laplacian Edge Detection",
            "Image Sharpening",
            "Histogram Equalization",
            "Binary Threshold",
            "Median Filter (Noise Reduction)"
        ]
    )
    
    # Filter parameters
    filter_params = {}
    if filter_type == "Gaussian Blur":
        filter_params['sigma'] = st.sidebar.slider("Blur Intensity (Sigma)", 0.1, 5.0, 1.0, 0.1)
        st.sidebar.latex(r"G(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}")
        
    elif filter_type == "Image Sharpening":
        filter_params['alpha'] = st.sidebar.slider("Sharpening Strength (Alpha)", 0.1, 3.0, 1.5, 0.1)
        st.sidebar.latex(r"S = Original + \alpha \times (Original - Blurred)")
        
    elif filter_type == "Binary Threshold":
        filter_params['threshold'] = st.sidebar.slider("Threshold Value", 0, 255, 127, 1)
        st.sidebar.latex(r"f(x,y) = \begin{cases} 255 & \text{if } pixel(x,y) > threshold \\ 0 & \text{otherwise} \end{cases}")
        
    elif filter_type == "Median Filter (Noise Reduction)":
        filter_params['kernel_size'] = st.sidebar.slider("Kernel Size", 3, 9, 3, 2)
    
    # Mathematical explanation for selected filter
    with st.sidebar.expander("Filter Mathematical Details"):
        if filter_type == "Grayscale (Standard)":
            st.markdown("Simple grayscale conversion using PIL's built-in method.")
        elif filter_type == "Grayscale (Weighted)":
            st.markdown("Weighted grayscale conversion using the formula:")
            st.latex(r"Y = 0.299 \times R + 0.587 \times G + 0.114 \times B")
        elif filter_type == "Sobel Edge Detection":
            st.markdown("Uses two 3x3 kernels to approximate gradients:")
            st.latex(r"G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}")
            st.latex(r"G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}")
            st.latex(r"G = \sqrt{G_x^2 + G_y^2}")
        elif filter_type == "Laplacian Edge Detection":
            st.markdown("Uses a Laplacian kernel for edge detection:")
            st.latex(r"Laplacian = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}")
        elif filter_type == "Histogram Equalization":
            st.markdown("Enhances contrast using the cumulative distribution function:")
            st.latex(r"T(r) = (L-1) \times CDF(r)")
            st.markdown("where L is the number of gray levels and CDF is the cumulative distribution function.")
    
    # Image upload section
    st.header("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPEG, PNG, etc.)", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"]
    )
    
    # Process the uploaded image
    if uploaded_file is not None:
        try:
            # Display processing status
            with st.spinner('Processing image...'):
                # Load the image
                original_image = Image.open(uploaded_file)
                
                # Apply selected filter
                if filter_type == "Grayscale (Standard)":
                    processed_image = convert_to_grayscale(original_image)
                    filter_description = "Standard grayscale conversion"
                    
                elif filter_type == "Grayscale (Weighted)":
                    processed_image = convert_to_grayscale_weighted(original_image)
                    filter_description = "Weighted grayscale using human perception formula"
                    
                elif filter_type == "Gaussian Blur":
                    processed_image = apply_gaussian_blur(original_image, sigma=filter_params.get('sigma', 1.0))
                    filter_description = f"Gaussian blur with sigma={filter_params.get('sigma', 1.0)}"
                    
                elif filter_type == "Sobel Edge Detection":
                    processed_image = apply_sobel_edge_detection(original_image)
                    filter_description = "Sobel edge detection highlighting intensity gradients"
                    
                elif filter_type == "Laplacian Edge Detection":
                    processed_image = apply_laplacian_edge_detection(original_image)
                    filter_description = "Laplacian edge detection highlighting areas of rapid intensity change"
                    
                elif filter_type == "Image Sharpening":
                    processed_image = apply_sharpening(original_image, alpha=filter_params.get('alpha', 1.5))
                    filter_description = f"Image sharpening with strength alpha={filter_params.get('alpha', 1.5)}"
                    
                elif filter_type == "Histogram Equalization":
                    processed_image = apply_histogram_equalization(original_image)
                    filter_description = "Histogram equalization for contrast enhancement"
                    
                elif filter_type == "Binary Threshold":
                    processed_image = apply_binary_threshold(original_image, threshold=filter_params.get('threshold', 127))
                    filter_description = f"Binary threshold with value={filter_params.get('threshold', 127)}"
                    
                elif filter_type == "Median Filter (Noise Reduction)":
                    processed_image = apply_median_filter(original_image, kernel_size=filter_params.get('kernel_size', 3))
                    filter_description = f"Median filter with kernel size={filter_params.get('kernel_size', 3)}"
                
                # Display filter information
                st.subheader(f"Applied Filter: {filter_type}")
                st.markdown(filter_description)
                
                # Create two columns for displaying images
                col1, col2 = st.columns(2)
                
                # Display original image
                with col1:
                    st.subheader("Original Image")
                    st.image(original_image, use_column_width=True)
                
                # Display processed image
                with col2:
                    st.subheader("Processed Image")
                    st.image(processed_image, use_column_width=True)
                
                # Save the processed image for download
                buffer = io.BytesIO()
                processed_image.save(buffer, format="PNG")
                
                # Download button
                st.download_button(
                    label=f"Download Processed Image ({filter_type})",
                    data=buffer.getvalue(),
                    file_name=f"processed_image_{filter_type.lower().replace(' ', '_')}.png",
                    mime="image/png"
                )
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please upload a valid image file.")
    else:
        # Display example/placeholder when no image is uploaded
        st.info("Upload an image to apply image processing filters.")
        
        # Example of what the app does
        st.subheader("How it works")
        st.markdown("""
        1. Upload a color image using the file uploader above
        2. Select a filter from the sidebar
        3. Adjust filter parameters if available
        4. Both original and processed images are displayed
        5. Download the processed version if desired
        """)
    
    # Additional information
    with st.expander("Supported Image Formats"):
        st.markdown("""
        - JPEG (.jpg, .jpeg)
        - PNG (.png)
        - BMP (.bmp)
        - TIFF (.tiff)
        """)
    
    # Technical information about DIP
    with st.expander("About Digital Image Processing"):
        st.markdown("""
        Digital Image Processing (DIP) involves the use of computer algorithms to perform operations on digital images. 
        Key areas include:
        
        1. **Image Enhancement**: Improving the quality or visibility of images
        2. **Edge Detection**: Identifying boundaries within an image
        3. **Noise Reduction**: Removing unwanted artifacts from images
        4. **Segmentation**: Partitioning images into meaningful regions
        5. **Morphological Processing**: Operations based on shape
        
        These techniques are used in various fields including medical imaging, remote sensing, 
        computer vision, and pattern recognition.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Digital Image Processing Application - Apply Various Image Filters")

if __name__ == "__main__":
    main()
