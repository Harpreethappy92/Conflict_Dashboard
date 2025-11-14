# conflict_dashboard_with_leader_follower_option.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from folium.plugins import Draw, HeatMap
from streamlit_folium import st_folium

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Traffic Conflict Dashboard", layout="wide")
st.title("🚦 Traffic Conflict Dashboard")

# -----------------------------
# -----------------------------
# FILE SELECTION FROM GITHUB
# -----------------------------
import requests

st.sidebar.header("📂 Select conflict dataset from GitHub")

# GitHub configuration
GITHUB_USER = "Harpreethappy92"      # replace with your GitHub username
GITHUB_REPO = "SampleData"          # replace with your repo name
#GITHUB_BRANCH = "main"             # branch name
#GITHUB_FOLDER = "data"             # folder containing your CSVs

# Fetch file list from GitHub API
url = f"https://github.com/Harpreethappy92/SampleData"
response = requests.get(url)

if response.status_code != 200:
    st.error("Failed to fetch file list from GitHub. Check repo/folder/branch settings.")
    st.stop()

files = response.json()
csv_files = [f["name"] for f in files if f["name"].endswith(".csv")]

if not csv_files:
    st.warning("No CSV files found in the specified GitHub folder.")
    st.stop()

selected_file = st.sidebar.selectbox("Choose a conflict data CSV", csv_files)

# Load the selected file
raw_url = f"https://github.com/Harpreethappy92/SampleData"

try:
    df = pd.read_csv(raw_url)
    st.success(f"✅ Loaded '{selected_file}' with {len(df)} rows and {len(df.columns)} columns")
except Exception as e:
    st.error(f"Failed to load CSV from GitHub: {e}")
    st.stop()
















# FILE UPLOAD
# -----------------------------
#uploaded_file = st.file_uploader("📂 Upload your conflict data (CSV or Excel)", type=["csv", "xlsx"])

#if not uploaded_file:
   # st.info("👆 Please upload your conflict dataset to begin.")
  #  st.stop()

# Read file
#if uploaded_file.name.endswith(".csv"):
   # df = pd.read_csv(uploaded_file)
#else:
   # df = pd.read_excel(uploaded_file)

#st.success(f"✅ Loaded data with {len(df)} rows and {len(df.columns)} columns")

# -----------------------------
# ROAD USER MAP (UI only)
# -----------------------------
road_user_map = {
    1: "Commercial vehcile",
    2: "Bus",
    3: "Passenger car",
    4: "Pedestrian",
    5: "Bicycle",
    6: "Motorbike",
    7: "Ute/Pickup truck",
    8: "Van",
    9: "Child",
    10: "Adult",
    11: "Elderly",
    12: "Dog",
    13: "Pram",
    14: "Scooter",
    15: "Skateboard",
    16: "Wheelchair",
    17: "Mobility scooters",
    18: "Individual Dog",
    19: "Articulated truck",
    20: "Rigid truck",
    21: "Double trailer",
    22: "Triple trailer",
    23: "Car/Ute with trailer",
    24: "E-scooters"
}
# reverse mapping for converting back to numeric codes
road_user_reverse_map = {v: k for k, v in road_user_map.items()}

# -----------------------------
# TIME CONVERSION
# -----------------------------
time_col = "time" if "time" in df.columns else ("video" if "video" in df.columns else None)
if time_col:
    df["datetime"] = pd.to_datetime(((df[time_col] / 86400000) + 25569), unit="D", origin="1899-12-30")
    df["Day"] = df["datetime"].dt.date
    df["Hour"] = df["datetime"].dt.hour
else:
    st.warning("⚠️ No 'time' or 'video' column detected — Day/Hour plots will not be available.")

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔍 Filter Options")

# Indicator mode: TTC / PET / BOTH
indicator_mode = st.sidebar.selectbox("Select indicator mode", ["ttc", "pet", "both"], index=0)

# Thresholds: show appropriate inputs
if indicator_mode == "ttc":
    ttc_threshold = st.sidebar.number_input("TTC threshold", value=3.0, step=0.1)
    pet_threshold = None
elif indicator_mode == "pet":
    pet_threshold = st.sidebar.number_input("PET threshold", value=3.0, step=0.1)
    ttc_threshold = None
else:  # both
    ttc_threshold = st.sidebar.number_input("TTC threshold", value=3.0, step=0.1)
    pet_threshold = st.sidebar.number_input("PET threshold", value=3.0, step=0.1)

filter_type = st.sidebar.radio("Filter Type (applies to selected indicator(s))", ["Below Threshold", "Above Threshold"])

# Leader-follower option — show if pet included
apply_leader_follower = False
if indicator_mode in ("pet", "both"):
    apply_leader_follower = st.sidebar.checkbox("Apply 'Correct Leader–Follower' Filter (PET only)", value=False)

# Ensure expected columns exist
expected_cols = [
    "RoadUser1_type", "RoadUser2_type",
    "RoadUser1_direction", "RoadUser2_direction",
    "Encounter_type"
]
for col in expected_cols:
    if col not in df.columns:
        df[col] = np.nan

# -----------------------------
# BUILD UI LABEL LISTS (no modification of df)
# -----------------------------
# For road user lists: build from unique numeric codes present in the dataset,
# map to labels for display. Keep the original numeric df unmodified.
def build_label_list_from_numeric(col):
    if col not in df.columns:
        return []
    uniques = sorted(pd.Series(df[col].dropna().unique()))
    labels = [road_user_map.get(int(x), str(int(x))) if (not pd.isna(x) and str(x).strip() != "") else "" for x in uniques]
    return labels

ru1_list = build_label_list_from_numeric("RoadUser1_type")
ru2_list = build_label_list_from_numeric("RoadUser2_type")

# Directions and encounter types (use raw values as before)
dir1_list = sorted(df["RoadUser1_direction"].dropna().unique())
dir2_list = sorted(df["RoadUser2_direction"].dropna().unique())
encounter_list = sorted(df["Encounter_type"].dropna().unique())

# Sidebar multiselects (showing labels for road users)
ru1_selected_labels = st.sidebar.multiselect("Road User 1 Type", ru1_list, default=ru1_list)
ru2_selected_labels = st.sidebar.multiselect("Road User 2 Type", ru2_list, default=ru2_list)
dir1_selected = st.sidebar.multiselect("RoadUser1 Direction", dir1_list, default=dir1_list)
dir2_selected = st.sidebar.multiselect("RoadUser2 Direction", dir2_list, default=dir2_list)
encounter_selected = st.sidebar.multiselect("Encounter Type", encounter_list, default=encounter_list)

# Convert selected labels back to numeric codes for filtering (if label isn't found, try converting to int)
def labels_to_numeric(labels):
    nums = []
    for lbl in labels:
        if lbl in road_user_reverse_map:
            nums.append(road_user_reverse_map[lbl])
        else:
            # attempt to parse label as integer (in case label is numeric string)
            try:
                nums.append(int(lbl))
            except Exception:
                # skip unconvertible labels
                pass
    return nums

ru1_selected = labels_to_numeric(ru1_selected_labels)
ru2_selected = labels_to_numeric(ru2_selected_labels)

# -----------------------------
# BUILD BASE FILTERED DF (ru/dir/encounter)
# -----------------------------
base_mask = pd.Series(True, index=df.index)

if ru1_selected:
    base_mask &= df["RoadUser1_type"].isin(ru1_selected)
if ru2_selected:
    base_mask &= df["RoadUser2_type"].isin(ru2_selected)
if dir1_selected:
    base_mask &= df["RoadUser1_direction"].isin(dir1_selected)
if dir2_selected:
    base_mask &= df["RoadUser2_direction"].isin(dir2_selected)
if encounter_selected:
    base_mask &= df["Encounter_type"].isin(encounter_selected)

base_df = df[base_mask].copy()

# -----------------------------
# INDICATOR-SPECIFIC FILTERS
# -----------------------------
ttc_df = pd.DataFrame()
pet_df = pd.DataFrame()

def apply_threshold(df_in, col, threshold, filter_type):
    if col not in df_in.columns:
        return pd.DataFrame()
    tmp = df_in[df_in[col].notna()].copy()
    if filter_type == "Below Threshold":
        return tmp[tmp[col] <= threshold].copy()
    else:
        return tmp[tmp[col] >= threshold].copy()

if indicator_mode in ("ttc", "both"):
    if "ttc" in base_df.columns:
        ttc_df = apply_threshold(base_df, "ttc", ttc_threshold, filter_type)
    else:
        ttc_df = pd.DataFrame()

if indicator_mode in ("pet", "both"):
    if "pet" in base_df.columns:
        pet_df = apply_threshold(base_df, "pet", pet_threshold, filter_type)
    else:
        pet_df = pd.DataFrame()

# -----------------------------
# APPLY LEADER–FOLLOWER TO PET (optional)
# -----------------------------
if apply_leader_follower and not pet_df.empty:
    # leader–follower logic uses the ORIGINAL numeric codes
    PEDESTRIAN_CODES = [4, 9, 10, 11, 12, 13, 15, 16, 17, 18, 24]
    BICYCLE_CODES = [5]
    VRU_CODES = PEDESTRIAN_CODES + BICYCLE_CODES

    def determine_true_conflict(row):
        if row.get("Encounter_type") != "VRU":
            return 0
        leader_type = row.get("RoadUser1_type")
        follower_type = row.get("RoadUser2_type")
        if leader_type in PEDESTRIAN_CODES:
            return 1
        if leader_type in BICYCLE_CODES and follower_type in PEDESTRIAN_CODES:
            return 0
        if leader_type in BICYCLE_CODES and follower_type not in VRU_CODES:
            return 1
        if leader_type not in VRU_CODES and follower_type in VRU_CODES:
            return 0
        if (leader_type in VRU_CODES) and (follower_type in VRU_CODES):
            return 1
        return 0

    pet_df = pet_df.copy()
    pet_df["True_conflict"] = pet_df.apply(determine_true_conflict, axis=1)
    pet_df = pet_df[pet_df["True_conflict"] == 1].copy()

    st.sidebar.success("✅ Applied 'Correct Leader–Follower' PET logic")

# -----------------------------
# COMBINE FILTERED RESULTS BASED ON MODE
# -----------------------------
if indicator_mode == "ttc":
    filtered_df = ttc_df.copy()
elif indicator_mode == "pet":
    filtered_df = pet_df.copy()
else:  # both
    filtered_df = pd.concat([ttc_df, pet_df], ignore_index=True).drop_duplicates().reset_index(drop=True)

filtered_df = filtered_df.copy()

st.markdown(f"**Filtered conflicts: {len(filtered_df)}**")

if filtered_df.empty:
    st.warning("No data after applying filters. Adjust thresholds/filters and try again.")
    st.stop()

# -----------------------------
# PREPARE PLOTTING FIELDS
# -----------------------------
if "datetime" in filtered_df.columns:
    filtered_df["Day_dt"] = pd.to_datetime(filtered_df["Day"])
else:
    try:
        filtered_df["Day_dt"] = pd.to_datetime(filtered_df["Day"])
    except Exception:
        filtered_df["Day_dt"] = pd.NaT

filtered_df["Weekday"] = filtered_df["Day_dt"].dt.day_name()

day_count = filtered_df.groupby("Day_dt").size().reset_index(name="Conflict count").sort_values("Day_dt")
day_count["Weekday"] = day_count["Day_dt"].dt.day_name()

hour_count = filtered_df.groupby("Hour").size().reset_index(name="Conflict count")
all_hours = pd.DataFrame({"Hour": range(24)})
hour_count = all_hours.merge(hour_count, on="Hour", how="left").fillna(0)

color_map = {
    "Monday": "blue",
    "Tuesday": "blue",
    "Wednesday": "blue",
    "Thursday": "blue",
    "Friday": "blue",
    "Saturday": "orange",
    "Sunday": "orange"
}

# -----------------------------
# DESCRIPTIVE STATISTICS + PLOTS
# -----------------------------
st.subheader("📊 Descriptive Statistics")

col1, col2 = st.columns(2)
with col1:
    if not day_count.empty:
        day_count['Weekday'] = pd.Categorical(
            day_count['Weekday'],
            categories=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
            ordered=True
        )
        day_count['color'] = day_count['Weekday'].map(color_map)
        fig_day = px.bar(
            day_count,
            x='Day_dt',
            y='Conflict count',
            title="Conflicts count per Day"
        )
        fig_day.update_xaxes(dtick=1)
        fig_day.update_layout(
            title_font_size=24,
            xaxis_title_font_size=18,
            yaxis_title_font_size=18,
            xaxis_tickfont_size=14,
            yaxis_tickfont_size=14,
            xaxis=dict(
                tickmode="array",
                tickvals=list(day_count["Day_dt"]),
                ticktext=[d.strftime("%d-%b (%a)") for d in day_count["Day_dt"]],
                tickangle=-45
            ),
            xaxis_title="Date",
            yaxis_title="Conflict count",
            bargap=0.05
        )
        fig_day.update_yaxes(
            tickfont=dict(size=14),
            title_font=dict(size=18),
            showgrid=True,
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor="gray"
        )
        st.plotly_chart(fig_day, use_container_width=True)
    else:
        st.info("No day data to plot.")

with col2:
    fig_hour = px.bar(hour_count, x="Hour", y="Conflict count", title="Conflicts count per Hour")
    fig_hour.update_xaxes(dtick=1)
    fig_hour.update_layout(
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis_tickfont_size=14,
        yaxis_tickfont_size=14)
    st.plotly_chart(fig_hour, use_container_width=True)

# Indicator distribution(s)
st.subheader("Indicator Distribution")
if indicator_mode == "ttc":
    if "ttc" in filtered_df.columns:
        st.plotly_chart(px.histogram(filtered_df, x="ttc", nbins=30, title="TTC distribution"), use_container_width=True)
    else:
        st.warning("TTC column not present.")
elif indicator_mode == "pet":
    if "pet" in filtered_df.columns:
        st.plotly_chart(px.histogram(filtered_df, x="pet", nbins=30, title="PET distribution"), use_container_width=True)
    else:
        st.warning("PET column not present.")
else:
    r1, r2 = st.columns(2)
    with r1:
        if "ttc" in filtered_df.columns and filtered_df["ttc"].notna().any():
            st.plotly_chart(px.histogram(filtered_df[filtered_df["ttc"].notna()], x="ttc", nbins=30, title="TTC distribution"), use_container_width=True)
        else:
            st.info("No TTC values in filtered set.")
    with r2:
        if "pet" in filtered_df.columns and filtered_df["pet"].notna().any():
            st.plotly_chart(px.histogram(filtered_df[filtered_df["pet"].notna()], x="pet", nbins=30, title="PET distribution"), use_container_width=True)
        else:
            st.info("No PET values in filtered set.")

if indicator_mode == "both":
    st.subheader("Indicator Summary (Both)")
    rows = []
    if "ttc" in filtered_df.columns and filtered_df["ttc"].notna().any():
        df_ttc = filtered_df[filtered_df["ttc"].notna()]
        ttc_stats = df_ttc["ttc"].describe().to_frame().T
        ttc_stats.insert(0, "indicator", "ttc")
        rows.append(ttc_stats)
    if "pet" in filtered_df.columns and filtered_df["pet"].notna().any():
        df_pet = filtered_df[filtered_df["pet"].notna()]
        pet_stats = df_pet["pet"].describe().to_frame().T
        pet_stats.insert(0, "indicator", "pet")
        rows.append(pet_stats)
    if rows:
        st.dataframe(pd.concat(rows, ignore_index=True).set_index("indicator"))
    else:
        st.info("No TTC/PET numeric data found in filtered dataset.")

# -----------------------------
# HEATMAP SECTION (combined filtered_df)
# -----------------------------
st.subheader("🌍 Conflict Heatmap")

def detect_latlon(df_local):
    lat_candidates = [c for c in df_local.columns if c.lower() in ("lat", "latitude", "lat_dd", "y", "lat_deg")]
    lon_candidates = [c for c in df_local.columns if c.lower() in ("lon", "lng", "longitude", "long", "x", "lon_deg")]
    lat = lat_candidates[0] if lat_candidates else None
    lon = lon_candidates[0] if lon_candidates else None
    return lat, lon

lat_col, lon_col = detect_latlon(filtered_df)

if lat_col is None or lon_col is None:
    st.info("No lat/lon auto-detected. Please select manually.")
    lat_col = st.selectbox("Latitude column", options=[None] + list(filtered_df.columns))
    lon_col = st.selectbox("Longitude column", options=[None] + list(filtered_df.columns))

if lat_col and lon_col and lat_col in filtered_df.columns and lon_col in filtered_df.columns:
    center_lat = float(filtered_df[lat_col].mean())
    center_lon = float(filtered_df[lon_col].mean())
else:
    center_lat, center_lon = -27.47, 153.02

zoom = st.slider("Zoom level", 12, 22, 17)

m_heat = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None, max_zoom=22)
folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                 attr="Google Satellite", name="Google Satellite", max_zoom=22).add_to(m_heat)

if lat_col in filtered_df.columns and lon_col in filtered_df.columns:
    pts = filtered_df[[lat_col, lon_col]].dropna().values.tolist()
    if pts:
        HeatMap(pts, radius=10, blur=15, max_zoom=22).add_to(m_heat)
folium.LayerControl().add_to(m_heat)
st.components.v1.html(m_heat._repr_html_(), height=600, scrolling=True)

# -----------------------------
# OPTIONAL ROI FILTER (after heatmap)
# -----------------------------
st.subheader("🟩 Optional: Draw Region of Interest (ROI)")

m_draw = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None, max_zoom=22)
folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                 attr="Google Satellite", name="Google Satellite", max_zoom=22).add_to(m_draw)

Draw(draw_options={"polygon": True, "rectangle": True, "polyline": False,
                   "circle": False, "marker": False, "circlemarker": False},
     edit_options={"edit": True, "remove": True}).add_to(m_draw)
folium.LayerControl().add_to(m_draw)

draw_result = st_folium(m_draw, returned_objects=["all_drawings", "last_active_drawing"], height=600, width="100%")

roi_filtered = pd.DataFrame()
if draw_result and draw_result.get("last_active_drawing") and lat_col in filtered_df.columns and lon_col in filtered_df.columns:
    roi = draw_result["last_active_drawing"]
    if "type" in roi and roi["type"] == "FeatureCollection":
        polygons = [shape(f["geometry"]) for f in roi["features"]]
    else:
        polygons = [shape(roi["geometry"])] if "geometry" in roi else [shape(roi)]
    roi_shape = unary_union(polygons)

    roi_filtered = filtered_df[filtered_df.apply(lambda r: Point(r[lon_col], r[lat_col]).within(roi_shape), axis=1)].copy()
    st.success(f"✅ {len(roi_filtered)} conflicts within ROI.")

    if not roi_filtered.empty:
        m_roi = folium.Map(location=[float(roi_filtered[lat_col].mean()), float(roi_filtered[lon_col].mean())],
                           zoom_start=zoom, tiles=None, max_zoom=22)
        folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                         attr="Google Satellite", name="Google Satellite", max_zoom=22).add_to(m_roi)
        HeatMap(roi_filtered[[lat_col, lon_col]].values.tolist(), radius=10, blur=15).add_to(m_roi)
        folium.LayerControl().add_to(m_roi)
        st.components.v1.html(m_roi._repr_html_(), height=500, scrolling=True)

# -----------------------------
# DOWNLOAD SECTION
# -----------------------------
st.subheader("💾 Download Data")

download_choice = st.radio(
    "Select which dataset to download:",
    ["Filtered data (no ROI)", "ROI-filtered data"],
    index=0
)

if download_choice == "ROI-filtered data" and not roi_filtered.empty:
    data_to_download = roi_filtered
    filename = f"filtered_conflicts_{indicator_mode}_ROI.csv"
else:
    data_to_download = filtered_df
    filename = f"filtered_conflicts_{indicator_mode}.csv"

# Export original numeric columns only (ensure no UI label fields are present)
# If you want to explicitly control exported columns, you can set export_cols list below.
export_df = data_to_download.copy()

# Drop any accidental UI label columns (defensive)
export_df = export_df.drop(columns=["RoadUser1_label", "RoadUser2_label"], errors="ignore")

if not export_df.empty:
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Selected Data",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )
else:
    st.warning("No data available for download.")
