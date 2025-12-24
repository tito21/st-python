import numpy as np
import cairo
import scipy.ndimage as ndi

from tractography import bilinear_interpolate
from bezier import bezier_point


def simple_brush(context, curve, color, width=1.0):
    context.set_line_join(cairo.LineJoin.ROUND)
    context.set_line_width(width)
    context.set_source_rgba(color[0], color[1], color[2], 1)
    context.move_to(curve[0][1], curve[0][0])
    context.curve_to(
        curve[1][1], curve[1][0], curve[2][1], curve[2][0], curve[3][1], curve[3][0]
    )
    context.stroke()


def angle_brush(
    context, curve, color, width=1.0, line_width=0.1, scale="constant", num_segments=3
):
    context.set_line_width(line_width)
    context.set_source_rgba(color[0], color[1], color[2], 1)

    t = np.linspace(0, 1, num_segments + 1)

    Q1 = 3.0 * (curve[1:] - curve[:-1])

    p = bezier_point(3, curve, t)
    d_p = bezier_point(2, Q1, t)

    if scale == "constant":
        d_p /= np.linalg.norm(d_p, axis=1, keepdims=True) + 1e-6
        d_p *= 0.5 * width
    elif isinstance(scale, (int, float)):
        norm = np.linalg.norm(d_p, axis=1, keepdims=True) + 1e-6
        d_p /= norm
        norm = norm / np.max(norm)
        d_p *= scale * norm
        d_p *= 0.5 * width
    else:
        raise ValueError("scale must be 'constant' or a float")

    for i in range(num_segments):
        pi = p[i]
        d_pi = d_p[i]
        p1 = pi + np.array([-d_pi[1], d_pi[0]])
        p2 = pi - np.array([-d_pi[1], d_pi[0]])

        context.move_to(p1[1], p1[0])
        context.line_to(p2[1], p2[0])
        context.stroke()


def img_brush(context, curve, color, width=1.0, image=None, jitter=0.1):
    if image is None:
        raise ValueError("image must be provided for img_brush")

    surface = context.get_target()
    buf = surface.get_data()
    assert surface.get_format() == cairo.FORMAT_ARGB32, "Surface format must be ARGB32"
    data = np.ndarray(
        shape=(surface.get_height(), surface.get_width()), dtype=np.uint32, buffer=buf
    )

    t = np.linspace(0, 1, 30)

    # Q1 = 3.0 * (curve[1:] - curve[:-1])

    p = bezier_point(3, curve, t)
    # d_p = bezier_point(2, Q1, t)

    # d_p /= np.linalg.norm(d_p, axis=1, keepdims=True) + 1e-6
    # d_p *= 0.5 * width
    color_argb = (
        (color[0] * 255).astype(np.uint32) << 16
        | (color[1] * 255).astype(np.uint32) << 8
        | (color[2] * 255).astype(np.uint32)
    )
    image = ndi.zoom(
        image, (width / image.shape[0], width / image.shape[1], 1), order=0
    )
    for i in range(len(t)):
        pi = p[i] + width * np.random.uniform(-jitter, jitter, size=2)
        # d_pi = d_p[i]
        # angle = np.arctan2(d_pi[0], d_pi[1])
        # image = ndi.rotate(image, np.degrees(angle), reshape=True)
        h, w, _ = image.shape
        x_start = int(pi[1] - w // 2)
        y_start = int(pi[0] - h // 2)
        x_end = x_start + w
        y_end = y_start + h
        if x_start < 0 or y_start < 0 or x_end > data.shape[1] or y_end > data.shape[0]:
            continue
        region = data[y_start:y_end, x_start:x_end]
        alpha = image[..., 3]
        inv_alpha = 1.0 - alpha
        brush_argb = (
            ((1 - image[:, :, 0]) * (color_argb & 0x00FF0000)).astype(np.uint32)
            & 0x00FF0000
            | ((1 - image[:, :, 1]) * (color_argb & 0x0000FF00)).astype(np.uint32)
            & 0x0000FF00
            | ((1 - image[:, :, 2]) * (color_argb & 0x000000FF)).astype(np.uint32)
            & 0x000000FF
        ) | 0xFF000000

        region[:] = (
            (
                (region & 0x00FF0000) * inv_alpha + (brush_argb & 0x00FF0000) * alpha
            ).astype(np.uint32)
            & 0x00FF0000
            | (
                (region & 0x0000FF00) * inv_alpha + (brush_argb & 0x0000FF00) * alpha
            ).astype(np.uint32)
            & 0x0000FF00
            | (
                (region & 0x000000FF) * inv_alpha + (brush_argb & 0x000000FF) * alpha
            ).astype(np.uint32)
            & 0x000000FF
        ) | 0xFF000000


def line_brush(context, curve, color, width=1.0, num_segments=10):
    context.set_line_join(cairo.LineJoin.ROUND)
    context.set_line_width(0.5 * (width / num_segments))
    context.set_source_rgba(color[0], color[1], color[2], 1)
    angle = np.arctan2(curve[3][0] - curve[0][0], curve[3][1] - curve[0][1])
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    for r in range(-num_segments // 2, num_segments // 2):
        offset = (r / num_segments) * width
        offest_x = offset * cos_a + np.random.uniform(-0.5 * (width / num_segments), 0.5 * (width / num_segments))
        offest_y = offset * sin_a + np.random.uniform(-0.5 * (width / num_segments), 0.5 * (width / num_segments))
        context.move_to(curve[0][1] + offest_x, curve[0][0] + offest_y)
        context.curve_to(
            curve[1][1] + offest_x,
            curve[1][0] + offest_y,
            curve[2][1] + offest_x,
            curve[2][0] + offest_y,
            curve[3][1] + offest_x,
            curve[3][0] + offest_y,
        )
        context.stroke()


def draw_tracts(tracts, image, context, brush):
    for tract in tracts:
        for curve in tract:
            mid_point = curve.shape[0] // 2
            color = bilinear_interpolate(image, curve[mid_point])
            brush(context, curve, color)
