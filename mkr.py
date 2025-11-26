import cv2
import numpy as np

SLOPE = 10.0
INTERCEPT = 950

DISAPPEAR_FRAMES = 10
MIN_LIFETIME_FRAMES = 15

GLASS_BLUR_KSIZE = 9
GLASS_BLUR_SIGMA = 3
GLASS_THRESH = 30

MIN_CLUSTER_OVERLAP = 0.5
AREA_CHANGE_THRESH = 0.25

EDGE_MARGIN = 35

def is_near_edge(x, y, w, h, img_w, img_h, margin=EDGE_MARGIN):
    return (
        x < margin
        or y < margin
        or x + w > img_w - margin
        or y + h > img_h - margin
    )
def match_object(x, y, objects):
    for obj in objects:
        if obj.get("seen_this_frame"):
            continue
        ox, oy, ow, oh = obj["bbox"]
        dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
        if dist < 30:
            return obj
    return None
def refine_bbox_with_red(x, y, w, h, red_mask, pad=10):
    h_img, w_img = red_mask.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(x + w, w_img)
    y1 = min(y + h, h_img)

    if x1 <= x0 or y1 <= y0:
        return x, y, w, h

    roi = red_mask[y0:y1, x0:x1]
    ys, xs = np.where(roi > 0)

    if len(xs) == 0:
        return x, y, w, h

    min_x = max(xs.min() - pad, 0)
    max_x = min(xs.max() + pad, roi.shape[1] - 1)

    new_x = x0 + min_x
    new_w = max_x - min_x + 1

    return new_x, y0, new_w, (y1 - y0)
def refine_bbox_with_glass_clusters(x, y, w, h, labels, stats, pad=5,
                                    min_cluster_overlap=MIN_CLUSTER_OVERLAP):
    h_img, w_img = labels.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(x + w, w_img)
    y1 = min(y + h, h_img)

    if x1 <= x0 or y1 <= y0:
        return x, y, w, h

    roi_labels = labels[y0:y1, x0:x1]
    cluster_labels = np.unique(roi_labels)
    cluster_labels = cluster_labels[cluster_labels > 0]

    if len(cluster_labels) == 0:
        return x, y, w, h

    union_x0 = x0
    union_y0 = y0
    union_x1 = x1
    union_y1 = y1

    for lbl in cluster_labels:
        cx = stats[lbl, cv2.CC_STAT_LEFT]
        cy = stats[lbl, cv2.CC_STAT_TOP]
        cw = stats[lbl, cv2.CC_STAT_WIDTH]
        ch = stats[lbl, cv2.CC_STAT_HEIGHT]
        cluster_area = stats[lbl, cv2.CC_STAT_AREA]

        if cluster_area <= 0:
            continue

        c_x0 = cx
        c_y0 = cy
        c_x1 = cx + cw
        c_y1 = cy + ch

        inter_x0 = max(x0, c_x0)
        inter_y0 = max(y0, c_y0)
        inter_x1 = min(x1, c_x1)
        inter_y1 = min(y1, c_y1)

        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            continue

        intersect_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        overlap_ratio = intersect_area / float(cluster_area)

        if overlap_ratio < min_cluster_overlap:
            continue

        c_x0_pad = c_x0 - pad
        c_y0_pad = c_y0 - pad
        c_x1_pad = c_x1 + pad
        c_y1_pad = c_y1 + pad

        union_x0 = min(union_x0, c_x0_pad)
        union_y0 = min(union_y0, c_y0_pad)
        union_x1 = max(union_x1, c_x1_pad)
        union_y1 = max(union_y1, c_y1_pad)

    union_x0 = max(0, int(union_x0))
    union_y0 = max(0, int(union_y0))
    union_x1 = min(w_img, int(union_x1))
    union_y1 = min(h_img, int(union_y1))

    new_w = union_x1 - union_x0
    new_h = union_y1 - union_y0

    if new_w <= 0 or new_h <= 0:
        return x, y, w, h

    return union_x0, union_y0, new_w, new_h
def glass_black_filter(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_dark = np.array([90, 65, 0], dtype=np.uint8)
    upper_dark = np.array([140, 90, 120], dtype=np.uint8)

    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    k = GLASS_BLUR_KSIZE
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(mask_dark, (k, k), GLASS_BLUR_SIGMA)

    _, mask_bin = cv2.threshold(blurred, GLASS_THRESH, 255, cv2.THRESH_BINARY)

    result = np.zeros_like(frame)
    result[mask_bin > 0] = (255, 0, 0)

    return result, mask_bin
def main():
    video_path = "video.mov"
    cap = cv2.VideoCapture(video_path)

    new_width = 640
    new_height = 360

    prev_gray = None
    car_count = 0
    active_objects = []
    next_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (new_width, new_height))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            continue

        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for obj in active_objects:
            obj["seen_this_frame"] = False

        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 50, 50], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

        lower_red2 = np.array([170, 50, 50], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        red_only = cv2.bitwise_and(resized, resized, mask=red_mask)

        glass_view, glass_mask = glass_black_filter(resized)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            glass_mask, connectivity=8
        )

        overlay = resized.copy()
        overlay[glass_mask > 0] = (255, 0, 0)
        alpha = 0.0
        vis = cv2.addWeighted(overlay, alpha, resized, 1 - alpha, 0)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            if area < 50:
                continue
            if y < new_height // 3:
                continue

            min_area = SLOPE * y + INTERCEPT
            if area < min_area:
                continue

            x_ref, y_ref, w_ref, h_ref = refine_bbox_with_red(
                x, y, w, h, red_mask, pad=10
            )

            x_ref, y_ref, w_ref, h_ref = refine_bbox_with_glass_clusters(
                x_ref, y_ref, w_ref, h_ref, labels, stats, pad=5,
                min_cluster_overlap=MIN_CLUSTER_OVERLAP
            )

            new_area = w_ref * h_ref
            near_edge = is_near_edge(x_ref, y_ref, w_ref, h_ref, new_width, new_height)

            match = match_object(x_ref, y_ref, active_objects)

            if match:
                old_area = match.get("area", new_area)
                match["bbox"] = (x_ref, y_ref, w_ref, h_ref)
                match["frames_since_seen"] = 0
                match["seen_this_frame"] = True
                match["lifetime"] += 1

                if old_area > 0:
                    rel_change = abs(new_area - old_area) / float(old_area)
                    match["stable"] = rel_change < AREA_CHANGE_THRESH
                else:
                    match["stable"] = True
                match["area"] = new_area

                ratio = w_ref / float(h_ref) if h_ref > 0 else 1.0

                if match["stable"] and not near_edge:
                    match["aspect_sum"] = match.get("aspect_sum", 0.0) + ratio
                    match["aspect_count"] = match.get("aspect_count", 0) + 1
                    avg_ratio = match["aspect_sum"] / match["aspect_count"]
                    match["size_class"] = "small" if avg_ratio >= 0.95 else "medium+"

                if "size_class" not in match:
                    match["size_class"] = "small" if ratio >= 0.95 else "medium+"

                color = match["color"]
                obj_id = match["id"]
                stable_flag = match["stable"]
                size_class = match["size_class"]
            else:
                obj_id = next_id
                next_id += 1
                stable_flag = True
                ratio = w_ref / float(h_ref) if h_ref > 0 else 1.0
                size_class = "small" if ratio >= 1.0 else "medium+"

                if near_edge:
                    aspect_sum = 0.0
                    aspect_count = 0
                else:
                    aspect_sum = ratio
                    aspect_count = 1

                new_obj = {
                    "id": obj_id,
                    "bbox": (x_ref, y_ref, w_ref, h_ref),
                    "area": new_area,
                    "stable": stable_flag,
                    "aspect_sum": aspect_sum,
                    "aspect_count": aspect_count,
                    "size_class": size_class,
                    "frames_since_seen": 0,
                    "seen_this_frame": True,
                    "lifetime": 1,
                    "color": (0, 255, 0),
                }
                active_objects.append(new_obj)
                color = new_obj["color"]

            cv2.rectangle(vis, (x_ref, y_ref), (x_ref + w_ref, y_ref + h_ref), color, 2)
            text_pos = (x_ref, max(0, y_ref - 5))
            status_text = (
                f"{obj_id}: stable, {size_class}"
                if stable_flag
                else f"{obj_id}: not stable, {size_class}"
            )
            cv2.putText(
                vis,
                status_text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        for obj in active_objects[:]:
            if not obj["seen_this_frame"]:
                obj["frames_since_seen"] += 1
                if obj["frames_since_seen"] > DISAPPEAR_FRAMES:
                    if obj["lifetime"] >= MIN_LIFETIME_FRAMES:
                        car_count += 1
                    active_objects.remove(obj)

        cv2.putText(
            vis,
            f"Cars: {car_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Video (with glass overlay)", vis)
        cv2.imshow("Diff", thresh)
        cv2.imshow("Red Filter (Lights)", red_only)
        cv2.imshow("Glass Blue (Debug)", glass_view)

        prev_gray = gray.copy()

        if cv2.waitKey(30) == ord("q"):
            break

    car_count += len(active_objects)
    print("Final count:", car_count)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
