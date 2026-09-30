"""
Perspective and geometric correction module.
Provides automatic deskewing (rotation correction) and 4-point perspective transformation
for industrial equipment tags, label plates, and CCTV camera angles.
"""

import logging
from typing import Tuple, Optional, List
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PerspectiveCorrector:
    """Handles angle deskewing and 4-point perspective transformation."""

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float, background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Rotates an image by a specific angle around its center, automatically adjusting
        the canvas size so that no corners are clipped.

        Args:
            image: Input image array.
            angle: Angle in degrees (positive = counterclockwise, negative = clockwise).
            background_color: Fill color for new border areas.

        Returns:
            np.ndarray: Rotated image.
        """
        if abs(angle) < 0.1:
            return image.copy()

        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        # Compute rotation matrix
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute new bounding dimensions of the rotated image
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to account for translation
        rot_mat[0, 2] += (new_w / 2.0) - center[0]
        rot_mat[1, 2] += (new_h / 2.0) - center[1]

        border_val = background_color if len(image.shape) == 3 else background_color[0]
        rotated = cv2.warpAffine(
            image,
            rot_mat,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val
        )
        return rotated

    @classmethod
    def calculate_skew_angle(cls, image: np.ndarray, max_angle: float = 45.0) -> float:
        """
        Calculates the dominant skew angle of text lines using minAreaRect on binary text pixels
        or Hough line transform.

        Args:
            image: Input image.
            max_angle: Maximum allowable angle to correct (avoids false 90-degree flips).

        Returns:
            float: Skew angle in degrees.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Invert and threshold: text becomes white foreground (255), background becomes black (0)
        # Apply Otsu threshold on inverted grayscale
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Dilate text horizontally to merge characters into horizontal word blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # Find all foreground coordinates
        pts = cv2.findNonZero(dilated)
        if pts is None or len(pts) < 50:
            return 0.0

        # Calculate minimum bounding rectangle
        rect = cv2.minAreaRect(pts)
        angle = rect[-1]

        # OpenCV minAreaRect returns angle in range [-90, 0) or [0, 90) depending on version
        if angle < -45.0:
            angle = -(90.0 + angle)
        elif angle > 45.0:
            angle = 90.0 - angle
        else:
            angle = -angle

        # If angle exceeds safety threshold, reject to avoid wild rotations
        if abs(angle) > max_angle:
            logger.debug(f"Computed skew angle {angle:.2f}° exceeds max_angle {max_angle}°, ignoring")
            return 0.0

        logger.debug(f"Detected skew angle: {angle:.2f}°")
        return angle

    @classmethod
    def deskew(cls, image: np.ndarray, max_angle: float = 45.0) -> Tuple[np.ndarray, float]:
        """
        Calculates skew angle and rotates image to level the text lines.

        Returns:
            Tuple[np.ndarray, float]: (Deskewed image, applied angle).
        """
        angle = cls.calculate_skew_angle(image, max_angle=max_angle)
        if abs(angle) >= 0.5:
            deskewed = cls.rotate_image(image, angle)
            return deskewed, angle
        return image.copy(), 0.0

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """
        Orders 4 coordinates in consistent order:
        [top-left, top-right, bottom-right, bottom-left].
        """
        pts = pts.reshape((4, 2))
        rect = np.zeros((4, 2), dtype="float32")

        # Top-left has smallest sum (x + y), bottom-right has largest sum (x + y)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right has smallest difference (y - x), bottom-left has largest difference (y - x)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    @classmethod
    def four_point_transform(cls, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Applies 4-point perspective warp transformation.

        Args:
            image: Input image.
            pts: Array of 4 (x, y) coordinates representing quadrilateral corners.

        Returns:
            np.ndarray: Top-down warped image.
        """
        rect = cls.order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute width of the new warped image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # Compute height of the new warped image
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        if max_width <= 10 or max_height <= 10:
            return image.copy()

        # Destination coordinates for flat rectangular view
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        # Compute perspective transform matrix & warp
        m = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, m, (max_width, max_height), flags=cv2.INTER_CUBIC)
        return warped

    @classmethod
    def find_label_contour(
        cls,
        image: np.ndarray,
        min_area_ratio: float = 0.05
    ) -> Optional[np.ndarray]:
        """
        Finds the prominent 4-point quadrilateral contour representing an equipment nameplate,
        label tag, or display screen.

        Args:
            image: Input BGR image.
            min_area_ratio: Minimum ratio of image area that the contour must occupy.

        Returns:
            Optional[np.ndarray]: 4 corner points if found, else None.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]
        total_area = h * w
        min_area = total_area * min_area_ratio

        # Edge detection with bilateral smoothing
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edged = cv2.Canny(blurred, 50, 200)

        # Morphological close to bridge edge gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # Check if contour is a convex 4-point polygon
            if len(approx) == 4 and cv2.isContourConvex(approx):
                logger.info(f"Found 4-point label contour with area={area:.0f}px ({(area/total_area)*100:.1f}% of image)")
                return approx.reshape((4, 2))

        return None

    @classmethod
    def correct_perspective_if_detected(
        cls,
        image: np.ndarray,
        min_area_ratio: float = 0.05
    ) -> Tuple[np.ndarray, bool]:
        """
        Attempts to detect a 4-point label tag and warp it into a flat top-down perspective.

        Returns:
            Tuple[np.ndarray, bool]: (Resulting image, was_corrected_flag).
        """
        contour = cls.find_label_contour(image, min_area_ratio=min_area_ratio)
        if contour is not None:
            try:
                warped = cls.four_point_transform(image, contour)
                return warped, True
            except Exception as e:
                logger.warning(f"Perspective warp failed on detected contour: {e}")

        return image.copy(), False
