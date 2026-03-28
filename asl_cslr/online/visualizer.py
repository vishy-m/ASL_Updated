"""
Real-time visualization: skeleton overlay and recognized gloss display (§9).
"""

import cv2
import numpy as np

from asl_cslr.data.skeleton import (
    NUM_JOINTS,
    IDX_LEFT_SHOULDER,
    IDX_RIGHT_SHOULDER,
    IDX_MID_SHOULDERS,
)


# Body skeleton edges (connect canonical pose joint indices)
BODY_EDGES = [
    (0, IDX_MID_SHOULDERS),  # Nose → MidShoulders
    (IDX_LEFT_SHOULDER, IDX_LEFT_ELBOW := 3),
    (3, IDX_LEFT_WRIST := 5),
    (IDX_RIGHT_SHOULDER, IDX_RIGHT_ELBOW := 4),
    (4, IDX_RIGHT_WRIST := 6),
    (IDX_LEFT_SHOULDER, IDX_LEFT_HIP := 7),
    (IDX_RIGHT_SHOULDER, IDX_RIGHT_HIP := 8),
    (IDX_LEFT_SHOULDER, IDX_RIGHT_SHOULDER),
    (7, 8),
]

# Hand finger chains (offsets relative to hand base: 10 for left, 31 for right)
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],    # Thumb
    [0, 5, 6, 7, 8],    # Index
    [0, 9, 10, 11, 12],  # Middle
    [0, 13, 14, 15, 16], # Ring
    [0, 17, 18, 19, 20], # Pinky
]


def draw_skeleton(
    frame: np.ndarray,
    raw_joints: np.ndarray,
    frame_width: int | None = None,
    frame_height: int | None = None,
    body_color: tuple = (255, 230, 0),         # Cyan BGR
    left_hand_color: tuple = (180, 105, 255),  # Pink BGR
    right_hand_color: tuple = (0, 140, 255),   # Orange BGR
    joint_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw a premium skeleton overlay on a frame with drop-shadows and vibrant colors.

    Args:
        frame: BGR image to draw on (modified in place).
        raw_joints: (52, 2) raw MediaPipe skeleton coordinates in [0..1] range.
        frame_width: Optional frame width for mapping.
        frame_height: Optional frame height for mapping.
        body_color: BGR color for body edges.
        left_hand_color: BGR color for left hand.
        right_hand_color: BGR color for right hand.
        joint_radius: Dot radius.
        line_thickness: Line width.

    Returns:
        Frame with skeleton drawn on it.
    """
    h, w = frame.shape[:2]
    fw = frame_width or w
    fh = frame_height or h

    def to_pixel(xy):
        """Convert [0..1] normalized coords directly to pixel coords."""
        if np.isnan(xy[0]) or np.isnan(xy[1]):
            return None
        return (int(xy[0] * fw), int(xy[1] * fh))

    # Helper to draw a glowing drop-shadowed line
    def draw_edge(p1, p2, color):
        if p1 and p2:
            # Drop shadow (black, thick)
            cv2.line(frame, p1, p2, (0, 0, 0), line_thickness + 3, cv2.LINE_AA)
            # Core neon line
            cv2.line(frame, p1, p2, color, line_thickness, cv2.LINE_AA)

    # Draw body edges
    for i, j in BODY_EDGES:
        draw_edge(to_pixel(raw_joints[i]), to_pixel(raw_joints[j]), body_color)

    # Draw hands
    for hand_offset, color in [(10, left_hand_color), (31, right_hand_color)]:
        for chain in FINGER_CHAINS:
            for k in range(len(chain) - 1):
                idx1 = hand_offset + chain[k]
                idx2 = hand_offset + chain[k + 1]
                draw_edge(to_pixel(raw_joints[idx1]), to_pixel(raw_joints[idx2]), color)

    # Draw joint dots (beautiful two-tone: white core, colored border)
    for i in range(NUM_JOINTS):
        px = to_pixel(raw_joints[i])
        if px:
            color = body_color
            if 10 <= i < 31:
                color = left_hand_color
            elif 31 <= i < 52:
                color = right_hand_color
            # Outer dark ring
            cv2.circle(frame, px, joint_radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            # Colored border ring
            cv2.circle(frame, px, joint_radius, color, -1, cv2.LINE_AA)
            # White core spec
            cv2.circle(frame, px, max(1, joint_radius - 2), (255, 255, 255), -1, cv2.LINE_AA)

    return frame


def draw_glosses(
    frame: np.ndarray,
    glosses: list[str],
    position: tuple = (20, 50),
    font_scale: float = 1.2,
    color: tuple = (255, 255, 255),
    bg_color: tuple = (0, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw recognized glosses as text on the frame.

    Args:
        frame: BGR image to draw on.
        glosses: List of gloss strings to display.
        position: (x, y) top-left corner for text.
        font_scale: Font size.
        color: Text color (BGR).
        bg_color: Background rectangle color.
        thickness: Text thickness.

    Returns:
        Frame with glosses drawn.
    """
    text = " ".join(glosses[-5:]) if glosses else "..."

    # Draw background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - 5, y - th - 10), (x + tw + 5, y + baseline + 5), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame
