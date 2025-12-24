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

from draw import draw_tracts, simple_brush, angle_brush, img_brush, line_brush
from tractography import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Non photo realistic rendering of images by structural tensor"
    )
    parser.add_argument("path", type=pathlib.Path, help="Path to the input image")
    parser.add_argument("output", type=pathlib.Path, help="Path to the output image")
    parser.add_argument(
        "--params", type=pathlib.Path, default=None, help="Path to the parameters file"
    )
    parser.add_argument(
        "--orientation-vector",
        default="structural",
        choices=["structural", "gradient"],
        help="Orientation vector to use",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
        help="Standard deviation of the avenging gaussian filter for the structural tensor",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Standard deviation of the derivative of gaussian filter for the structural tensor",
    )
    parser.add_argument(
        "--method",
        default="continuous",
        choices=["grid", "continuous"],
        help="Rendering method to use",
    )
    parser.add_argument(
        "--brush",
        default="simple",
        choices=["simple", "angle", "img", "line"],
        help="Brush type to use",
    )
    parser.add_argument(
        "--brush-img",
        type=pathlib.Path,
        default=None,
        help="Path to the brush image (only for img brush)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Save intermediate debug images"
    )
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
                "color_threshold": 50,
            }
        ]
    else:
        with open(params_path, "r") as f:
            settings = json.load(f)
        return settings


def get_brush(brush_type, width, brush_img_path):
    match brush_type:
        case "simple":
            brush = partial(simple_brush, width=width)
        case "angle":
            brush = partial(
                angle_brush,
                width=width,
                line_width=width / 10,
                num_segments=5,
                scale=0.75,
            )
        case "img":
            brush_image = io.imread(brush_img_path)
            brush_image = brush_image.astype(np.float32) / 255.0
            brush = partial(img_brush, width=width, image=brush_image)
        case "line":
            brush = partial(line_brush, width=width, num_segments=15)
    return brush


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
    brush=None,
):
    start_points = [
        (i, j)
        for i in range(0, image.shape[0], grid_size)
        for j in range(0, image.shape[1], grid_size)
    ]
    random.shuffle(start_points)
    for start_point in tqdm.tqdm(start_points):
        # start_point = (
        #     np.clip(
        #         start_point[0] + np.random.randint(0, grid_size // 2),
        #         grid_size // 2,
        #         image.shape[0] - grid_size // 2,
        #     ),
        #     np.clip(
        #         start_point[1] + np.random.randint(0, grid_size // 2),
        #         grid_size // 2,
        #         image.shape[1] - grid_size // 2,
        #     ),
        # )
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
            max_index = np.argmax(
                valid_mask[
                    start_point[0] : start_point[0] + grid_size,
                    start_point[1] : start_point[1] + grid_size,
                ]
            )
            pos_x, pos_y = np.unravel_index(max_index, (grid_size, grid_size))
            x0 = start_point[0] + pos_x
            y0 = start_point[1] + pos_y

            # x0 = np.clip(start_point[0] + np.random.randint(-grid_size // 2, grid_size // 2), 0, image.shape[0] - grid_size)
            # y0 = np.clip(start_point[1] + np.random.randint(-grid_size // 2, grid_size // 2), 0, image.shape[1] - grid_size)

            # x0 = start_point[0] + grid_size // 2
            # y0 = start_point[1] + grid_size // 2
            tract = compute_tract(
                ode_system, (x0, y0), length_lines, min_length, tolerance=width
            )

            draw_tracts([tract], image, context, brush)


def render_continuous(
    context,
    surface,
    image,
    orientation,
    valid_mask,
    num_lines=10000,
    mask_threshold=0.5,
    color_difference_threshold=100,
    length_lines=1000.0,
    min_length=1.0,
    width=2.0,
    brush=None,
):

    lines_drawn = 0
    target = np.ndarray(
        buffer=surface.get_data(),
        shape=(image.shape[0], image.shape[1]),
        dtype=np.uint32,
    )
    target = np.stack([((target >> 16) & 0xFF), ((target >> 8) & 0xFF), (target & 0xFF)], axis=-1).astype(np.uint8)

    color_difference = np.linalg.norm(image - target, axis=-1)
    error = np.mean(color_difference)

    bar = tqdm.tqdm(total=num_lines)
    while lines_drawn < num_lines and error > color_difference_threshold:

        index = np.random.choice(color_difference.size, p=color_difference.flatten() / np.sum(color_difference))
        x0, y0 = np.unravel_index(index, color_difference.shape)

        # max_index = np.unravel_index(np.argmax(color_difference), color_difference.shape)
        # x0, y0 = max_index

        # x0 = np.random.randint(0, image.shape[0])
        # y0 = np.random.randint(0, image.shape[1])

        ode_system = ODESystem(orientation, valid_mask, mask_threshold)

        target = np.ndarray(
            buffer=surface.get_data(),
            shape=(image.shape[0], image.shape[1]),
            dtype=np.uint32,
        )
        target = np.stack([((target >> 16) & 0xFF), ((target >> 8) & 0xFF), (target & 0xFF)], axis=-1).astype(np.uint8)

        color_difference = np.linalg.norm(image - target, axis=-1)
        error = np.mean(color_difference)
        tract = compute_tract(
            ode_system, (x0, y0), length_lines, min_length, tolerance=width
        )
        draw_tracts([tract], image, context, brush)

        lines_drawn += 1
        bar.update(1)
        bar.set_description(f"Lines drawn: {lines_drawn}, Color diff: {error:.2f}")
    bar.close()


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

    for i, setting in enumerate(settings):
        print(f"Rendering layer {i+1}/{len(settings)}")
        # Compute the structural tensor
        J11, J22, J12 = compute_structural_tensor(
            gaussian(image_gray, sigma=setting["sigma"]),
            rho=args.rho,
            sigma=args.sigma,
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

        match args.method:
            case "grid":
                render_grid(
                    context,
                    surface,
                    image,
                    orientation,
                    coh,
                    mask_threshold=0.5,
                    color_difference_threshold=setting.get("color_threshold", 50),
                    grid_size=int(setting["width"] / 2),
                    length_lines=setting["length_lines"],
                    min_length=setting["min_length"],
                    width=setting["width"],
                    brush=get_brush(args.brush, setting["width"], args.brush_img),
                )
            case "continuous":
                render_continuous(
                    context,
                    surface,
                    image,
                    orientation,
                    coh,
                    num_lines=setting["num_lines"],
                    mask_threshold=0.5,
                    color_difference_threshold=setting.get("color_threshold", 50),
                    length_lines=setting["length_lines"],
                    min_length=setting["min_length"],
                    width=setting["width"],
                    brush=get_brush(args.brush, setting["width"], args.brush_img),
                )

        print("Rendering completed.")
        if args.debug:
            debug_path = args.output.parent / f"debug_layer_{i+1}.png"
            surface.write_to_png(debug_path)
            print(f"Debug image saved to {debug_path}")

    surface.write_to_png(args.output)


if __name__ == "__main__":
    main()
