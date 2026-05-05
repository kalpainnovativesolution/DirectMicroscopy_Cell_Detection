import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re
import os
import hashlib
from pathlib import Path
from ultralytics import YOLO
import gdown


# ---------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Cell Detection App",
    page_icon="🔬",
    layout="wide"
)

st.title("Cell Detection with YOLOv11")

st.markdown(
    """
    Upload two image folders. The app will automatically sort images by filename, 
    select the first 25 images from each folder, detect cells, display results, 
    and calculate Somatic Cells/ml.
    """
)


# ---------------------------------------------------------------------
# Config Helpers
# ---------------------------------------------------------------------
def get_secret_value(key, default=""):
    """
    Safely read value from Streamlit secrets.
    Works locally and on Streamlit Cloud.
    """
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# ---------------------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------------------
st.sidebar.header("Settings")

default_model_drive_url = get_secret_value("MODEL_DRIVE_URL", "")

model_drive_url = st.sidebar.text_input(
    "Google Drive Model Link",
    value=default_model_drive_url,
    help="Paste the public Google Drive link of your YOLO .pt model file."
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05
)

show_labels = st.sidebar.checkbox("Show Labels on App Display", value=True)
show_conf = st.sidebar.checkbox("Show Confidence Scores on App Display", value=True)

box_color = st.sidebar.color_picker("Bounding Box Color", "#00FF00")

img_size = 640
max_files_per_group = 25
allowed_extensions = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}


# ---------------------------------------------------------------------
# Model Download and Load
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def download_and_load_model(drive_url):
    """
    Downloads YOLO model from Google Drive and loads it.
    The downloaded model is cached by Streamlit.
    """

    if not drive_url or not drive_url.strip():
        raise ValueError("Google Drive model link is missing.")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    url_hash = hashlib.md5(drive_url.encode()).hexdigest()[:10]
    model_path = model_dir / f"yolo_model_{url_hash}.pt"

    if not model_path.exists() or model_path.stat().st_size == 0:
        downloaded_file = gdown.download(
            url=drive_url,
            output=str(model_path),
            quiet=False,
            fuzzy=True
        )

        if downloaded_file is None or not model_path.exists():
            raise RuntimeError(
                "Model download failed. Please check whether the Google Drive link is public."
            )

    return YOLO(str(model_path))


try:
    with st.spinner("Loading YOLO model..."):
        model = download_and_load_model(model_drive_url)
    st.sidebar.success("Model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def resize_image(image_np, size=(640, 640)):
    return cv2.resize(image_np, size, interpolation=cv2.INTER_AREA)


def natural_sort_key(text):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def get_source_folder_name(file_path):
    """
    Extract top-level folder name from uploaded file path.
    Example:
    sampleA/img1.jpg -> sampleA
    sampleA/sub/img1.jpg -> sampleA
    img1.jpg -> root_folder
    """
    normalized = file_path.replace("\\", "/").strip("/")
    parts = normalized.split("/")

    if len(parts) >= 2:
        return parts[0]

    return "root_folder"


def get_filename_only(file_path):
    return os.path.basename(file_path.replace("\\", "/"))


def filter_and_select_first_25(uploaded_files, max_count=25):
    valid_files = []

    for file in uploaded_files:
        ext = file.name.split(".")[-1].lower() if "." in file.name else ""
        if ext in allowed_extensions:
            valid_files.append(file)

    valid_files = sorted(valid_files, key=lambda x: natural_sort_key(x.name))
    selected_files = valid_files[:max_count]

    return selected_files, len(valid_files)


def read_uploaded_image(uploaded_file):
    uploaded_file.seek(0)

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if bgr is None:
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def draw_detections(image_np, results, box_hex, show_lbl=False, show_cf=False):
    output = image_np.copy()

    hex_color = box_hex.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    color_bgr = (b, g, r)

    detections_by_class = {}

    for result in results:
        boxes = result.boxes
        names = result.names

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = names[cls]

            detections_by_class[label] = detections_by_class.get(label, 0) + 1

            cv2.rectangle(output, (x1, y1), (x2, y2), color_bgr, 2)

            if show_lbl or show_cf:
                text_parts = []

                if show_lbl:
                    text_parts.append(label)

                if show_cf:
                    text_parts.append(f"{conf:.2f}")

                text = " ".join(text_parts)

                if text.strip():
                    (tw, th), _ = cv2.getTextSize(
                        text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        1
                    )

                    label_y1 = max(y1 - th - 8, 0)
                    label_y2 = max(y1, th + 8)

                    cv2.rectangle(
                        output,
                        (x1, label_y1),
                        (x1 + tw + 6, label_y2),
                        color_bgr,
                        -1
                    )

                    text_y = max(y1 - 4, th + 2)

                    cv2.putText(
                        output,
                        text,
                        (x1 + 2, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

    return output, detections_by_class


def process_single_image(uploaded_file):
    raw_original_rgb = read_uploaded_image(uploaded_file)

    if raw_original_rgb is None:
        return None

    source_folder_name = get_source_folder_name(uploaded_file.name)
    image_name_only = get_filename_only(uploaded_file.name)

    resized_rgb = resize_image(raw_original_rgb, size=(img_size, img_size))

    results = model.predict(
        source=resized_rgb,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=img_size,
        verbose=False
    )

    display_image, detections_by_class = draw_detections(
        resized_rgb,
        results,
        box_color,
        show_labels,
        show_conf
    )

    total_cells = sum(detections_by_class.values())

    return {
        "filename": image_name_only,
        "original_uploaded_path": uploaded_file.name,
        "source_folder_name": source_folder_name,
        "raw_original_rgb": raw_original_rgb,
        "resized_rgb": resized_rgb,
        "display_image": display_image,
        "detections_by_class": detections_by_class,
        "total_cells": total_cells
    }


def process_image_group(uploaded_files, group_name):
    processed_results = []
    group_class_counts = {}
    group_total_cells = 0

    for uploaded_file in uploaded_files:
        result = process_single_image(uploaded_file)

        if result is None:
            continue

        processed_results.append(result)
        group_total_cells += result["total_cells"]

        for cls_name, count in result["detections_by_class"].items():
            group_class_counts[cls_name] = group_class_counts.get(cls_name, 0) + count

    return {
        "group_name": group_name,
        "processed_results": processed_results,
        "group_class_counts": group_class_counts,
        "group_total_cells": group_total_cells
    }


def make_class_dataframe(class_counts, total_cells):
    if total_cells == 0:
        return pd.DataFrame(columns=["Class", "Count", "% of Total"])

    rows = []

    for cls_name, count in class_counts.items():
        percent = (count / total_cells) * 100
        rows.append((cls_name, count, f"{percent:.1f}%"))

    return pd.DataFrame(rows, columns=["Class", "Count", "% of Total"])


def make_image_summary_dataframe(processed_results):
    rows = []

    for i, result in enumerate(processed_results, start=1):
        rows.append({
            "S.No.": i,
            "Source Folder": result["source_folder_name"],
            "Image Name": result["filename"],
            "Total Cells": result["total_cells"]
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Uploaders
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("Upload Image Folders")

st.caption(
    "Choose folders here. The app will sort images by filename and process "
    "only the first 25 images from each folder."
)

left_to_right_files = st.file_uploader(
    "Select left to right images folder",
    type=list(allowed_extensions),
    accept_multiple_files="directory",
    key="left_to_right_folder_uploader",
    help="Select a folder. The app will sort images by filename and use the first 25 images."
)

top_to_bottom_files = st.file_uploader(
    "Select top to bottom images folder",
    type=list(allowed_extensions),
    accept_multiple_files="directory",
    key="top_to_bottom_folder_uploader",
    help="Select a folder. The app will sort images by filename and use the first 25 images."
)


# ---------------------------------------------------------------------
# Select First 25 Automatically
# ---------------------------------------------------------------------
selected_left_files = []
selected_top_files = []
total_left_uploaded_valid = 0
total_top_uploaded_valid = 0

if left_to_right_files:
    selected_left_files, total_left_uploaded_valid = filter_and_select_first_25(
        left_to_right_files,
        max_files_per_group
    )

if top_to_bottom_files:
    selected_top_files, total_top_uploaded_valid = filter_and_select_first_25(
        top_to_bottom_files,
        max_files_per_group
    )


# ---------------------------------------------------------------------
# Selected Image Summary
# ---------------------------------------------------------------------
if left_to_right_files or top_to_bottom_files:
    st.markdown("---")
    st.subheader("Selected Image Summary")

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("**Left to Right Folder**")
        st.write(f"Valid uploaded images found: {total_left_uploaded_valid}")
        st.write(f"Images selected for processing: {len(selected_left_files)}")

        if selected_left_files:
            left_folder_name = get_source_folder_name(selected_left_files[0].name)
            st.write(f"**Source folder name:** {left_folder_name}")

        if total_left_uploaded_valid > max_files_per_group:
            st.info(
                f"Only the first {max_files_per_group} images after filename sorting are being processed."
            )

        if selected_left_files:
            st.write("**Selected filenames:**")
            st.write([get_filename_only(file.name) for file in selected_left_files])

    with col_b:
        st.write("**Top to Bottom Folder**")
        st.write(f"Valid uploaded images found: {total_top_uploaded_valid}")
        st.write(f"Images selected for processing: {len(selected_top_files)}")

        if selected_top_files:
            top_folder_name = get_source_folder_name(selected_top_files[0].name)
            st.write(f"**Source folder name:** {top_folder_name}")

        if total_top_uploaded_valid > max_files_per_group:
            st.info(
                f"Only the first {max_files_per_group} images after filename sorting are being processed."
            )

        if selected_top_files:
            st.write("**Selected filenames:**")
            st.write([get_filename_only(file.name) for file in selected_top_files])


# ---------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------
if selected_left_files or selected_top_files:

    with st.spinner("Running inference on selected images..."):
        left_group_data = process_image_group(
            selected_left_files,
            "Left to Right"
        )

        top_group_data = process_image_group(
            selected_top_files,
            "Top to Bottom"
        )

    total_left_cells = left_group_data["group_total_cells"]
    total_top_cells = top_group_data["group_total_cells"]
    grand_total_cells = total_left_cells + total_top_cells

    # Somatic Cells/ml formula:
    # (Combined Total Cells / 50) * 333300
    somatic_cells_per_ml = (grand_total_cells / 50) * 333300

    combined_class_counts = {}

    for cls_name, count in left_group_data["group_class_counts"].items():
        combined_class_counts[cls_name] = combined_class_counts.get(cls_name, 0) + count

    for cls_name, count in top_group_data["group_class_counts"].items():
        combined_class_counts[cls_name] = combined_class_counts.get(cls_name, 0) + count

    st.markdown("---")
    st.subheader("Final Summation")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Left to Right Total Cells", total_left_cells)

    with m2:
        st.metric("Top to Bottom Total Cells", total_top_cells)

    with m3:
        st.metric("Combined Total Cells", grand_total_cells)

    with m4:
        st.metric("Somatic Cells/ml", f"{somatic_cells_per_ml:,.2f}")

    st.markdown("---")
    st.subheader("Combined Per-Class Breakdown")

    combined_df = make_class_dataframe(combined_class_counts, grand_total_cells)
    st.dataframe(combined_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Left to Right Group Summary")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Number of Processed Images", len(left_group_data["processed_results"]))

    with c2:
        st.metric("Total Cells Detected", total_left_cells)

    left_img_summary_df = make_image_summary_dataframe(left_group_data["processed_results"])

    if not left_img_summary_df.empty:
        st.markdown("**Per-image total cells**")
        st.dataframe(left_img_summary_df, use_container_width=True, hide_index=True)

    left_class_df = make_class_dataframe(
        left_group_data["group_class_counts"],
        total_left_cells
    )

    st.markdown("**Per-class breakdown**")
    st.dataframe(left_class_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Top to Bottom Group Summary")

    c3, c4 = st.columns(2)

    with c3:
        st.metric("Number of Processed Images", len(top_group_data["processed_results"]))

    with c4:
        st.metric("Total Cells Detected", total_top_cells)

    top_img_summary_df = make_image_summary_dataframe(top_group_data["processed_results"])

    if not top_img_summary_df.empty:
        st.markdown("**Per-image total cells**")
        st.dataframe(top_img_summary_df, use_container_width=True, hide_index=True)

    top_class_df = make_class_dataframe(
        top_group_data["group_class_counts"],
        total_top_cells
    )

    st.markdown("**Per-class breakdown**")
    st.dataframe(top_class_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Combined Image-wise Summary")

    combined_rows = []

    for i, result in enumerate(left_group_data["processed_results"], start=1):
        combined_rows.append({
            "Group": "Left to Right",
            "S.No.": i,
            "Source Folder": result["source_folder_name"],
            "Image Name": result["filename"],
            "Total Cells": result["total_cells"]
        })

    for i, result in enumerate(top_group_data["processed_results"], start=1):
        combined_rows.append({
            "Group": "Top to Bottom",
            "S.No.": i,
            "Source Folder": result["source_folder_name"],
            "Image Name": result["filename"],
            "Total Cells": result["total_cells"]
        })

    combined_image_summary_df = pd.DataFrame(combined_rows)

    if not combined_image_summary_df.empty:
        st.dataframe(combined_image_summary_df, use_container_width=True, hide_index=True)

    if left_group_data["processed_results"]:
        st.markdown("---")
        st.subheader("Left to Right Image Results")

        for idx, result in enumerate(left_group_data["processed_results"], start=1):
            st.markdown(f"### Image {idx}: {result['filename']}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    result["raw_original_rgb"],
                    caption=f"Original RGB - {result['filename']}",
                    use_container_width=True
                )

            with col2:
                st.image(
                    result["resized_rgb"],
                    caption="Resized RGB 640 x 640",
                    use_container_width=True
                )

            with col3:
                st.image(
                    result["display_image"],
                    caption=f"Detected Cells: {result['total_cells']}",
                    use_container_width=True
                )

            st.write(f"**Source folder:** {result['source_folder_name']}")
            st.write(f"**Total cells in this image:** {result['total_cells']}")

            if result["detections_by_class"]:
                per_img_df = make_class_dataframe(
                    result["detections_by_class"],
                    result["total_cells"]
                )

                st.dataframe(per_img_df, use_container_width=True, hide_index=True)
            else:
                st.info("No cells detected in this image.")

    if top_group_data["processed_results"]:
        st.markdown("---")
        st.subheader("Top to Bottom Image Results")

        for idx, result in enumerate(top_group_data["processed_results"], start=1):
            st.markdown(f"### Image {idx}: {result['filename']}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    result["raw_original_rgb"],
                    caption=f"Original RGB - {result['filename']}",
                    use_container_width=True
                )

            with col2:
                st.image(
                    result["resized_rgb"],
                    caption="Resized RGB 640 x 640",
                    use_container_width=True
                )

            with col3:
                st.image(
                    result["display_image"],
                    caption=f"Detected Cells: {result['total_cells']}",
                    use_container_width=True
                )

            st.write(f"**Source folder:** {result['source_folder_name']}")
            st.write(f"**Total cells in this image:** {result['total_cells']}")

            if result["detections_by_class"]:
                per_img_df = make_class_dataframe(
                    result["detections_by_class"],
                    result["total_cells"]
                )

                st.dataframe(per_img_df, use_container_width=True, hide_index=True)
            else:
                st.info("No cells detected in this image.")

else:
    st.info("Please upload one or both image folders to get started.")
