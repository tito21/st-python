import numpy as np
import cairo

from tractography import bilinear_interpolate
from bezier import bezier_point

def simple_brush(context, curve, color, width=1.0):
    context.set_line_join(cairo.LineJoin.ROUND)
    context.set_line_width(width)
    context.set_source_rgba(color[0], color[1], color[2], 1)
    context.move_to(curve[0][1], curve[0][0])
    context.curve_to(curve[1][1], curve[1][0], curve[2][1], curve[2][0], curve[3][1], curve[3][0])
    context.stroke()


def angle_brush(context, curve, color, width=1.0, line_width=0.1, scale="constant", num_segments=3):
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
        p1 =  pi + np.array([-d_pi[1], d_pi[0]])
        p2 =  pi - np.array([-d_pi[1], d_pi[0]])

        context.move_to(p1[1], p1[0])
        context.line_to(p2[1], p2[0])
        context.stroke()



def draw_tracts(tracts, image, context, brush):
    for tract in tracts:
        for curve in tract:
            mid_point = curve.shape[0] // 2
            color = bilinear_interpolate(image, curve[mid_point])
            brush(context, curve, color)

