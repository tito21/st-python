import pathlib
import argparse
import random
import json
from functools import partial

import numpy as np
import skimage.io as io
from skimage.filters import gaussian
import cairo
import tqdm.auto as tqdm

from draw import draw_tracts, simple_brush, angle_brush
from tractography import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Non photo realistic rendering of images by structural tensor"
    )
    parser.add_argument("path", type=pathlib.Path, help="Path to the input image")
    parser.add_argument("output", type=pathlib.Path, help="Path to the output image")
    parser.add_argument("--params", type=pathlib.Path, default=None, help="Path to the parameters file")
    parser.add_argument("--orientation-vector", default="structural", choices=["structural", "gradient"], help="Orientation vector to use")
    return parser.parse_args()


def get_settings(params_path):
    if params_path is None:
        return [
            {
                "sigma": 1.0,
                "width": 2.0,
                "num_lines": 10000,
                "length_lines": 100.0,
                "min_length": 10.0,
                'color_threshold': 50
            }
        ]
    else:
        with open(params_path, "r") as f:
            settings = json.load(f)
        return settings


def render(
    context,
    image,
    orientation,
    valid_mask,
    mask_threshold=0.5,
    num_lines=10000,
    length_lines=100000.0,
    width=2.0,
):
    # Compute the tractography
    tracts = []

    for _ in tqdm.trange(num_lines):
        ode_system = ODESystem(orientation, valid_mask, mask_threshold)

        start_point = (
            np.random.randint(0, image.shape[0]),
            np.random.randint(0, image.shape[1]),
        )
        tracts.append(compute_tract(ode_system, start_point, length_lines))

    print("Tractography computed.")

    draw_tracts(tracts, image, context, width)


def render_grid(
    context,
    surface,
    image,
    orientation,
    valid_mask,
    mask_threshold=0.5,
    color_difference_threshold=100,
    grid_size=20,
    length_lines=1000.0,
    min_length=1.0,
    width=2.0,
):
    # brush = partial(angle_brush, width=width, line_width=width / 10, num_segments=5, scale=0.75)
    brush = partial(simple_brush, width=width)
    start_points = [
        (i, j)
        for i in range(0, image.shape[0], grid_size)
        for j in range(0, image.shape[1], grid_size)
    ]
    random.shuffle(start_points)
    for start_point in tqdm.tqdm(start_points):
        ode_system = ODESystem(orientation, valid_mask, mask_threshold)
        image_region = image[
            start_point[0] - grid_size // 2 : start_point[0] + grid_size // 2,
            start_point[1] - grid_size // 2 : start_point[1] + grid_size // 2,
        ]
        target_region = np.ndarray(
            buffer=surface.get_data(),
            shape=(image.shape[0], image.shape[1]),
            dtype=np.uint32,
        )[
            start_point[0] - grid_size // 2 : start_point[0] + grid_size // 2,
            start_point[1] - grid_size // 2 : start_point[1] + grid_size // 2,
        ]
        target_region = np.stack(
            [
                ((target_region >> 16) & 0xFF),
                ((target_region >> 8) & 0xFF),
                (target_region & 0xFF),
            ],
            axis=-1,
        ).astype(np.uint8)

        color_difference = np.mean(
            np.linalg.norm(image_region - target_region, axis=-1)
        )
        if color_difference > color_difference_threshold:
            x0 = start_point[0] + grid_size // 2
            y0 = start_point[1] + grid_size // 2
            tract = compute_tract(
                ode_system, (x0, y0), length_lines, min_length, tolerance=width
            )

            draw_tracts([tract], image, context, brush)


def main():
    args = parse_args()
    print(f"Input image path: {args.path}")
    print(f"Output image path: {args.output}")

    settings = get_settings(args.params)

    image = io.imread(args.path)
    image = image.astype(np.float32) / 255.0
    image_gray = np.mean(image, axis=2) if image.ndim == 3 else image
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, image.shape[1], image.shape[0])
    context = cairo.Context(surface)
    context.set_source_rgba(1, 1, 1, 1)  # Set background to white
    context.paint()

    for setting in settings:

        # Compute the structural tensor
        J11, J22, J12 = compute_structural_tensor(
            gaussian(image_gray, sigma=setting["sigma"])
        )
        print("Structural tensor computed.")

        # Compute the eigenvalues and eigenvectors
        eigvals, eigvecs = compute_eigensystem(J11, J22, J12)
        print("Eigensystem computed.", eigvals.shape, eigvecs.shape)

        # Compute the gradient and normal orientation
        gradient_orientation = compute_gradient_normal_orientation(image_gray)
        gradient_magnitude = np.linalg.norm(gradient_orientation, axis=-1)
        gradient_orientation /= gradient_magnitude[..., np.newaxis] + 1e-10
        print(
            "Gradient normal orientation computed.",
            gradient_orientation.shape,
            np.min(gradient_magnitude),
            np.max(gradient_magnitude),
        )

        # Compute the coherence
        coh = coherence(eigvals)
        print("Coherence computed.")

        if args.orientation_vector == "structural":
            orientation = eigvecs[..., 0]
            print("Using structural orientation vector.")
        else:
            orientation = gradient_orientation
            print("Using gradient orientation vector.")

        render_grid(
            context,
            surface,
            image,
            orientation,
            coh,
            mask_threshold=0.5,
            color_difference_threshold=setting.get("color_threshold", 50),
            grid_size=int(setting["width"]),
            length_lines=setting["length_lines"],
            min_length=setting["min_length"],
            width=setting["width"],
        )
        # render_grid(context, surface, image, gradient_orientation, gradient_magnitude, mask_threshold=5, color_difference_threshold=100, grid_size=int(setting["width"]), length_lines=setting["length_lines"], min_length=setting["min_length"], width=setting["width"])

        # render(context, image, eigvecs[..., 0], coh, mask_threshold=0.5, num_lines=setting["num_lines"], length_lines=setting["length_lines"], width=setting["width"])
        # render(context, image, gradient_orientation, gradient_magnitude, mask_threshold=20.0, num_lines=setting["num_lines"], length_lines=setting["length_lines"], width=setting["width"])
        print("Rendering completed.")

    surface.write_to_png(args.output)


if __name__ == "__main__":
    main()
