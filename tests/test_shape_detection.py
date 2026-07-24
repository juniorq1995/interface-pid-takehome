import cv2
import numpy as np

from src.pid_extraction.shape_detection import detect_shapes


def _blank_canvas(size=(400, 400)):
    return np.full((*size, 3), 255, dtype=np.uint8)


def test_detect_shapes_happy_path_rectangle_and_circle():
    img = _blank_canvas()
    cv2.rectangle(img, (50, 50), (150, 200), (0, 0, 0), -1)
    cv2.circle(img, (300, 120), 40, (0, 0, 0), -1)

    shapes = detect_shapes(img)
    kinds = sorted(s.kind for s in shapes)
    assert kinds == ["circle", "rectangle"]


def test_detect_shapes_pump_glyph_uses_child_contour():
    img = _blank_canvas()
    cx, cy = 200, 200
    cv2.circle(img, (cx, cy), 45, (0, 0, 0), -1)
    triangle = np.array([[cx - 18, cy + 14], [cx + 18, cy + 14], [cx, cy - 20]])
    cv2.fillPoly(img, [triangle], (255, 255, 255))

    shapes = detect_shapes(img)
    assert len(shapes) == 1
    assert shapes[0].kind == "circle_triangle"


def test_detect_shapes_ignores_tiny_noise_edge_case():
    img = _blank_canvas()
    cv2.rectangle(img, (50, 50), (58, 58), (0, 0, 0), -1)  # ~64px area, below MIN_CONTOUR_AREA

    shapes = detect_shapes(img)
    assert shapes == []


def test_detect_shapes_blank_image_failure_case():
    img = _blank_canvas()
    shapes = detect_shapes(img)
    assert shapes == []
