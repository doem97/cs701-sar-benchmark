import json, zipfile

def write_submission(images, preds, size=512, out="submission.zip"):
    """images: from images.json; preds[file_name] = (label,
    boxes [x1, y1, x2, y2] on the model input, classes, scores)"""
    rows, dets = ["image_id,label"], []
    for im in images:
        label, boxes, classes, scores = preds[im["file_name"]]
        rows.append(f"{im['id']},{int(label)}")  # the integer id
        sx, sy = im["width"] / size, im["height"] / size
        for (x1, y1, x2, y2), c, s in zip(boxes, classes, scores):
            box = [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy]
            dets.append({"image_id": im["id"],
                         "category_id": int(c) + 1,  # label + 1
                         "bbox": [float(v) for v in box],
                         "score": float(s)})
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("classification.csv", "\n".join(rows) + "\n")
        z.writestr("detection.json", json.dumps(dets))
