import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, CheckButtons


def visualize(image=None, mask=None, bboxes=None, default_depth=110):
    if image is None and mask is None:
        raise ValueError("At least one of 'image' or 'mask' must be provided.")

    max_depth = max(image.shape[0], mask.shape[0]) if image is not None and mask is not None else \
        image.shape[0] if image is not None else mask.shape[0]

    if not (0 <= default_depth < max_depth):
        default_depth = 0

    if image is not None and mask is not None and image.shape[0] != mask.shape[0]:
        raise ValueError("Image and mask must have the same number of layers")

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2, left=0.2)

    img_display = ax.imshow(image[default_depth], cmap="gray") if image is not None else None
    mask_display = ax.imshow(mask[default_depth], cmap="jet", alpha=0.7) if mask is not None else None
    bbox_patches = []

    def create_bbox_patches(depth_slice):
        for patch in bbox_patches:
            patch.remove()
        bbox_patches.clear()

        if bboxes:
            for bbox in bboxes:
                x, y, z, width, height, depth = bbox
                if z <= depth_slice < (z + depth):
                    rect = Rectangle((x, y), width, height, edgecolor="red", facecolor="none", linewidth=2)
                    bbox_patches.append(rect)
                    ax.add_patch(rect)

    create_bbox_patches(default_depth)
    ax.set_title(f"Layer {default_depth}")
    ax.axis("off")

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax=ax_slider, label="Layer", valmin=0, valmax=max_depth - 1, valinit=default_depth, valstep=1)

    ax_check = plt.axes([0.02, 0.4, 0.1, 0.15])
    check_labels = ["Image"] if image is not None else []
    if mask is not None:
        check_labels.append("Mask")
    check = CheckButtons(ax=ax_check, labels=check_labels, actives=[True] * len(check_labels))

    def update_slider(val):
        layer = int(slider.val)
        if image is not None and img_display is not None:
            img_display.set_array(image[layer])
        if mask is not None and mask_display is not None:
            mask_display.set_array(mask[layer])
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


def load_npy_or_npz(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.npz'):
        return np.load(file_path)['arr_0']
    else:
        raise ValueError("Unsupported file format. Use .npy or .npz.")


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D image and mask data.")
    parser.add_argument('-i', '--image', type=str, help="Path to the image file (.npy or .npz)")
    parser.add_argument('-m', '--mask', type=str, help="Path to the mask file (.npy or .npz)")
    args = parser.parse_args()

    image = load_npy_or_npz(args.image) if args.image else None
    mask = load_npy_or_npz(args.mask) if args.mask else None

    visualize(image=image, mask=mask)


if __name__ == "__main__":
    main()