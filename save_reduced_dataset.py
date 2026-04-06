import pandas as pd

# ============================================
# Load original dataset
# ============================================
df = pd.read_excel("/home/topsoe/vrsh/streamlit-d/example/vessel_input_cleaned.xlsx")

# ============================================
# Keep only relevant columns + target
# ============================================
keep_columns = [
    "EquipmentType",
    "BubblePointTemperature",
    "StreamTemperature",
    "Pressure",
    "EquipmentName",
    "DewPointTemperature",
    "DewPointPressure",
    "VesselOrientation",  # target
]

df_reduced = df[keep_columns]

# ============================================
# Save to new file
# ============================================
output_path = "/home/topsoe/vrsh/streamlit-d/example/vessel_data_reduced.xlsx"
df_reduced.to_excel(output_path, index=False)

print(f"Original:  {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Reduced:   {df_reduced.shape[0]} rows, {df_reduced.shape[1]} columns")
print(f"Dropped:   {df.shape[1] - df_reduced.shape[1]} columns")
print(f"\nSaved to: {output_path}")
print(f"\nKept columns:")
for col in keep_columns:
    print(f"  ✅ {col}")