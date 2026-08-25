"""
Generate the short project PDF report requested in the assignment:
"Summarize your EDA findings and features introduced. Report your
training and performance information."

Pulls the live numbers from artifacts/metrics.json + model_metadata.json
(written by train_model.py) and, if present, artifacts/test_runs.log
(written by test_model.py) for the official test-set score. The EDA
summary and feature list are static text distilled from
notebooks/NYC_trip_duration_EDA.ipynb.

Run (after train_model.py, and after test_model.py if you have a score):
    python generate_report.py --artifacts-dir artifacts --out report.pdf
"""

import argparse
import json
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
)


EDA_FINDINGS = [
    "Target variable: <b>trip_duration</b> is strongly right-skewed; a log1p transform "
    "makes it close to normal, so the model is trained on log1p(trip_duration).",
    "Vendors: vendor 2 consistently shows longer average durations and contributes almost "
    "all of the extreme delay outliers seen against the OSRM fastest-route estimate.",
    "Time patterns: demand peaks in the evening (17:00-19:00) with a smaller morning peak, "
    "and dips sharply overnight; a late-January/early-February dip lines up with winter weather.",
    "Geography: pickups cluster heavily in Manhattan, with a clear secondary cluster at JFK; "
    "JFK trips are disproportionately represented among the long-delay outliers.",
    "Weather: snowfall and heavier precipitation days coincide with lower median trip speed "
    "and a visible drop in daily trip volume.",
    "OSRM comparison: the fastest-route travel-time estimate is a good baseline, but real "
    "trips regularly run longer; most of that gap is explained by vendor, and a meaningful "
    "chunk by airport queuing time (trip_duration includes wait time OSRM can't see).",
    "Data quality: a small number of trips (a few hundred out of ~1.4M) have durations that "
    "are implausible given the routing/distance data - these were capped/removed before "
    "modeling (see Outlier Filtering below).",
]

FEATURES_INTRODUCED = [
    ("Datetime parts", "hour, day_of_week, month, is_weekend, is_rush_hour - extracted from pickup_datetime."),
    ("Distance features", "haversine_distance_km and manhattan_distance_km between pickup/dropoff "
     "(log1p-transformed, since both are right-skewed), plus compass bearing."),
    ("Airport features", "great-circle distance from pickup/dropoff to JFK and LaGuardia, and boolean "
     "jfk_trip / lga_trip flags (within 2000m of either airport)."),
    ("Weather features", "daily rain, snowfall (s_fall), total precipitation (all_precip), snow depth "
     "(s_depth), max/min temperature, and has_rain / has_snow flags, merged onto each trip by pickup date."),
    ("OSRM features", "osrm_distance, osrm_travel_time, and osrm_num_steps - the fastest theoretical "
     "route's distance, travel time, and step count between pickup and dropoff, merged by trip id."),
]

OUTLIER_RULES = [
    "Coordinates restricted to the NYC bounding box (lat 40.50-41.00, lon -74.25 to -73.70).",
    "trip_duration capped below 22 hours and above 10 seconds.",
    "Implied speed (haversine distance / duration) capped below 100 km/h.",
    "Zero-distance trips are dropped unless they're both very short in distance (<=10m) and "
    "duration (<60s), which plausibly represents a real near-instant trip rather than bad GPS data.",
    "passenger_count > 0 only.",
]


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def metric_row(label, m):
    if m is None:
        return [label, "-", "-", "-", "-"]
    return [label, str(m.get("n_rows", "-")), f"{m['rmse']:.4f}", f"{m['mae']:.4f}", f"{m['r2']:.4f}"]


def build_report(artifacts_dir, out_path):
    metadata = load_json(os.path.join(artifacts_dir, "model_metadata.json"))
    metrics = load_json(os.path.join(artifacts_dir, "metrics.json"))

    test_metrics = None
    test_log_path = os.path.join(artifacts_dir, "test_runs.log")
    if os.path.exists(test_log_path):
        with open(test_log_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            last = lines[-1]
            if last.get("metrics"):
                m = last["metrics"]
                test_metrics = {
                    "n_rows": m["n_rows"],
                    "rmse": m["rmse_log1p"],
                    "mae": m["mae_log1p"],
                    "r2": m["r2"],
                }

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=9, textColor=colors.grey))
    title_style = styles["Title"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]

    story = []
    story.append(Paragraph("NYC Taxi Trip Duration Prediction", title_style))
    story.append(Paragraph("Ridge Regression Model - Project Report", styles["Heading3"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Task: predict total taxi-ride duration in NYC from pickup/dropoff coordinates, "
        "timestamps, and trip metadata, optimizing R2 on log1p(trip_duration).",
        body))
    story.append(Spacer(1, 14))

    story.append(Paragraph("1. EDA Findings", h2))
    story.append(ListFlowable(
        [ListItem(Paragraph(f, body)) for f in EDA_FINDINGS],
        bulletType="bullet",
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Outlier Filtering (applied to training data only)", h2))
    story.append(ListFlowable(
        [ListItem(Paragraph(f, body)) for f in OUTLIER_RULES],
        bulletType="bullet",
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Features Introduced", h2))
    feat_rows = [["Group", "Description"]] + [[name, desc] for name, desc in FEATURES_INTRODUCED]
    feat_table = Table(feat_rows, colWidths=[1.5 * inch, 4.7 * inch])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Model & Training Setup", h2))
    if metadata:
        setup_lines = [
            f"Model: Ridge, alpha = {metadata['alpha']} (fixed per assignment spec).",
            f"Target: log1p({metadata['target']}).",
            f"Feature count after one-hot encoding: {len(metadata['feature_columns'])}.",
            f"Categorical column(s) one-hot encoded: {', '.join(metadata['categorical_cols']) or 'none'}.",
            f"Train/held-out split: {int((1 - metadata['test_size']) * 100)}/"
            f"{int(metadata['test_size'] * 100)}, random_state={metadata['random_state']}.",
            f"External data merged in: weather ({'yes' if metadata.get('weather_path') else 'no'}), "
            f"OSRM ({'yes' if metadata.get('osrm_paths') else 'no'}).",
        ]
    else:
        setup_lines = ["No trained model found - run train_model.py first."]
    story.append(ListFlowable(
        [ListItem(Paragraph(l, body)) for l in setup_lines],
        bulletType="bullet",
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("5. Performance", h2))
    table_data = [["Split", "# rows", "RMSE (log1p)", "MAE (log1p)", "R2"]]
    if metrics:
        table_data.append(metric_row("Train", metrics.get("train")))
        table_data.append(metric_row("Held-out (from train.csv)", metrics.get("holdout")))
    else:
        table_data.append(["Train", "-", "-", "-", "-"])
        table_data.append(["Held-out (from train.csv)", "-", "-", "-", "-"])
    table_data.append(metric_row("Official test set", test_metrics))

    perf_table = Table(table_data, colWidths=[1.9 * inch, 0.8 * inch, 1.1 * inch, 1.1 * inch, 0.9 * inch])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(perf_table)

    if test_metrics is None:
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Official test-set row is blank until test_model.py is run once against the "
            "real (password-protected) test.csv, per the assignment's one-shot testing rule.",
            styles["Small"]))

    story.append(Spacer(1, 14))
    story.append(Paragraph("6. Notes", h2))
    story.append(Paragraph(
        "Ridge(alpha=1) was used as the fixed model per the assignment spec, rather than "
        "tuning across models/hyperparameters, so that most of the effort went into data "
        "understanding and feature engineering instead.",
        body))

    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                             leftMargin=0.7 * inch, rightMargin=0.7 * inch)
    doc.build(story)
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the short project PDF report.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--out", default="report.pdf")
    args = parser.parse_args()

    build_report(args.artifacts_dir, args.out)
