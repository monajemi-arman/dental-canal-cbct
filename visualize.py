import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, CheckButtons


def visualize(image=None, mask=None, bboxes=None, default_depth=110):
    if image is None and mask is None:
        raise ValueError("At least one of 'image' or 'mask' must be provided.")
    
    # Validate depth against available input dimensions
    max_depth = 0
    if image is not None:
        max_depth = max(max_depth, image.shape[0])
    if mask is not None:
        max_depth = max(max_depth, mask.shape[0])
    
    # Ensure default_depth is within valid range for both image and mask
    if not (0 <= default_depth < max_depth):
        default_depth = 0
        
    # Validate that image and mask have the same depth if both are provided
    if image is not None and mask is not None:
        if image.shape[0] != mask.shape[0]:
            raise ValueError("Image and mask must have the same number of layers")
    
    # Initialize figure and adjust layout
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2, left=0.2)
    
    # Create initial displays
    img_display = None
    mask_display = None
    bbox_patches = []
    
    # Initialize both displays with the same default_depth
    if image is not None:
        img_display = ax.imshow(image[default_depth], cmap="gray")
    if mask is not None:
        mask_display = ax.imshow(mask[default_depth], cmap="jet", alpha=0.7)
    
    def create_bbox_patches(depth_slice):
        # Clear existing patches
        for patch in bbox_patches:
            patch.remove()
        bbox_patches.clear()
        
        if bboxes:
            for bbox in bboxes:
                x, y, z, width, height, depth = bbox
                # Only show bounding boxes that intersect with the current depth slice
                if z <= depth_slice < (z + depth):
                    rect = Rectangle(
                        (x, y),
                        width,
                        height,
                        edgecolor="red",
                        facecolor="none",
                        linewidth=2,
                    )
                    bbox_patches.append(rect)
                    ax.add_patch(rect)
    
    # Create initial bbox patches
    create_bbox_patches(default_depth)
    ax.set_title(f"Layer {default_depth}")
    ax.axis("off")
    
    # Add slider with the correct range based on the data
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="Layer",
        valmin=0,
        valmax=max_depth - 1,
        valinit=default_depth,
        valstep=1,
    )
    
    # Add checkboxes
    ax_check = plt.axes([0.02, 0.4, 0.1, 0.15])
    check_labels = []
    if image is not None:
        check_labels.append("Image")
    if mask is not None:
        check_labels.append("Mask")
    check = CheckButtons(
        ax=ax_check, labels=check_labels, actives=[True] * len(check_labels)
    )
    
    def update_slider(val):
        layer = int(slider.val)
        if image is not None and img_display is not None:
            img_display.set_array(image[layer])
        if mask is not None and mask_display is not None:
            mask_display.set_array(mask[layer])
        # Update bounding boxes for current layer
        create_bbox_patches(layer)
        ax.set_title(f"Layer {layer}")
        fig.canvas.draw_idle()
    
    def update_visibility(label):
        if label == "Image" and img_display is not None:
            img_display.set_visible(not img_display.get_visible())
        elif label == "Mask" and mask_display is not None:
            mask_display.set_visible(not mask_display.get_visible())
        fig.canvas.draw_idle()
    
    slider.on_changed(update_slider)
    check.on_clicked(update_visibility)
    plt.show()