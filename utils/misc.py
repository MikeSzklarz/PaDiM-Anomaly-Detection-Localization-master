import numpy as np


def denormalize_image_for_display(tensor_image):
    """Reverses ImageNet normalization for display.

    Expects a CHW numpy array or tensor-like that can be transposed.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    denormalized_img = (
        ((tensor_image.transpose(1, 2, 0) * std) + mean) * 255.0
    ).astype(np.uint8)
    return denormalized_img


def feature_factory_from_blob_df(blob_df):
    """Convert a raw per-blob dataframe into per-image features for ML.

    Parameters
    ----------
    blob_df : pandas.DataFrame
        DataFrame with one row per blob. Expected columns (at minimum):
        - image_filename or image_name or image_idx
        - area_pixels
        - max_intensity
        - quadrant (optional)
        - image_gt_label (optional)

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix (one row per image)
    y : pandas.Series
        True image-level labels (if available) aligned to X.index; may contain NaN
    feature_names : list[str]
        Ordered list of feature column names for reproducible column ordering

    Notes
    -----
    Missing numeric values are filled with 0. The function will create an
    "image_name" column (basename without extension) if needed and group by it.
    """
    import os
    import pandas as pd

    # Accept either a DataFrame or a path (string)
    if isinstance(blob_df, str):
        blob_df = pd.read_csv(blob_df)

    df = blob_df.copy()

    # Ensure we have an image_name column to group on
    if "image_name" not in df.columns:
        if "image_filename" in df.columns:
            df["image_name"] = (
                df["image_filename"]
                .fillna("")
                .apply(lambda p: os.path.splitext(os.path.basename(p))[0] if p else "")
            )
        else:
            # fallback to image_idx
            if "image_idx" in df.columns:
                df["image_name"] = df["image_idx"].apply(lambda i: f"img_{i}")
            else:
                # nothing to group by; raise
                raise ValueError(
                    "blob_df must contain at least one of: image_name, image_filename, image_idx"
                )

    # Ensure numeric conversions
    for col in ["area_pixels", "max_intensity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate stats per image for selected blob properties
    agg_funcs = {
        "area_pixels": ["sum", "mean", "max", "std", "count"],
        "max_intensity": ["sum", "mean", "max", "std"],
    }

    grouped = df.groupby("image_name").agg(agg_funcs)
    # flatten MultiIndex columns
    grouped.columns = ["_".join([col[0], col[1]]) for col in grouped.columns]

    # Rename consistent feature names
    rename_map = {}
    if "area_pixels_sum" in grouped.columns:
        rename_map["area_pixels_sum"] = "area_sum"
    if "area_pixels_mean" in grouped.columns:
        rename_map["area_pixels_mean"] = "area_mean"
    if "area_pixels_max" in grouped.columns:
        rename_map["area_pixels_max"] = "area_max"
    if "area_pixels_std" in grouped.columns:
        rename_map["area_pixels_std"] = "area_std"
    if "area_pixels_count" in grouped.columns:
        rename_map["area_pixels_count"] = "area_count"

    if "max_intensity_sum" in grouped.columns:
        rename_map["max_intensity_sum"] = "maxint_sum"
    if "max_intensity_mean" in grouped.columns:
        rename_map["max_intensity_mean"] = "maxint_mean"
    if "max_intensity_max" in grouped.columns:
        rename_map["max_intensity_max"] = "maxint_max"
    if "max_intensity_std" in grouped.columns:
        rename_map["max_intensity_std"] = "maxint_std"

    grouped = grouped.rename(columns=rename_map)

    # Pivot by quadrant: counts and area sums per quadrant
    quad_count = None
    quad_area = None
    if "quadrant" in df.columns:
        try:
            quad_count = (
                df.groupby(["image_name", "quadrant"]).size().unstack(fill_value=0)
            )
            quad_count.columns = [f"quad_count_{int(c)}" for c in quad_count.columns]
            quad_area = (
                df.groupby(["image_name", "quadrant"])["area_pixels"]
                .sum()
                .unstack(fill_value=0)
            )
            quad_area.columns = [f"quad_area_{int(c)}" for c in quad_area.columns]
        except Exception:
            quad_count = None
            quad_area = None

    parts = [grouped]
    if quad_count is not None:
        parts.append(quad_count)
    if quad_area is not None:
        parts.append(quad_area)

    if parts:
        features = parts[0].join(parts[1:], how="left") if len(parts) > 1 else parts[0]
    else:
        features = pd.DataFrame(index=df["image_name"].unique())

    # Fill NaNs with zeros for numeric stability
    features = features.fillna(0)

    # Collect labels per image if present
    y = None
    if "image_gt_label" in df.columns:
        y = df.groupby("image_name")["image_gt_label"].first().reindex(features.index)

    feature_names = list(features.columns)

    return features.reset_index(drop=False).set_index("image_name"), y, feature_names
