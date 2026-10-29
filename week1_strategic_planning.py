"""
YuvaIntern Logistics Data Analyst Internship — Week 1
Task: Strategic Planning and Data Exploration in Logistics

Purpose:
- Define a logistics delivery-performance scenario
- Identify business questions and KPIs
- Create a small illustrative shipment dataset for initial exploration
- Print a data overview and save a data dictionary and planning summary

IMPORTANT: This is simulated/illustrative data for learning only. It is not real
company, customer, or shipment information.

Install: python -m pip install pandas numpy
Run:     python week1_strategic_planning.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("week1_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Strategic scenario and business objective
SCENARIO = "A logistics provider wants to improve on-time delivery and understand shipment costs."
BUSINESS_OBJECTIVE = (
    "Use shipment-level data to understand delivery performance, identify possible delay factors, "
    "and establish KPIs for later analysis and predictive modelling."
)
BUSINESS_QUESTIONS = [
    "What percentage of shipments arrive on or before the planned delivery time?",
    "Which transport modes are associated with longer delivery times?",
    "How are distance, traffic, weather, and shipment weight related to delivery time?",
    "What is the average transport cost per shipment and per kilometre?",
    "Which fields need validation or cleaning before deeper analysis?",
]

# 2. Create a small, reproducible, simulated dataset for initial exploration
rng = np.random.default_rng(7)
n = 100
transport_modes = rng.choice(["Road", "Rail", "Air"], size=n, p=[0.65, 0.23, 0.12])
traffic_levels = rng.choice(["Low", "Medium", "High"], size=n, p=[0.30, 0.45, 0.25])
weather_conditions = rng.choice(["Clear", "Rain", "Storm"], size=n, p=[0.68, 0.27, 0.05])
distance = rng.integers(10, 701, size=n)
weight = rng.uniform(0.5, 40, size=n).round(1)

mode_hours_per_100km = pd.Series(transport_modes).map({"Road": 2.2, "Rail": 1.5, "Air": 0.55}).to_numpy()
traffic_delay = pd.Series(traffic_levels).map({"Low": 0.3, "Medium": 1.5, "High": 3.5}).to_numpy()
weather_delay = pd.Series(weather_conditions).map({"Clear": 0.0, "Rain": 1.2, "Storm": 3.0}).to_numpy()
actual_hours = np.maximum(
    0.5,
    distance / 100 * mode_hours_per_100km
    + traffic_delay + weather_delay + weight * 0.025
    + rng.normal(0, 1.2, size=n),
).round(2)
planned_hours = np.maximum(0.5, actual_hours + rng.normal(0.3, 1.4, size=n)).round(2)
mode_cost_per_km = pd.Series(transport_modes).map({"Road": 1.8, "Rail": 1.25, "Air": 4.8}).to_numpy()
transport_cost = np.maximum(5, distance * mode_cost_per_km + weight * 2 + rng.normal(0, 15, size=n)).round(2)

shipments = pd.DataFrame({
    "shipment_id": [f"SH{i:04d}" for i in range(1, n + 1)],
    "shipment_date": pd.date_range("2026-10-01", periods=n, freq="D"),
    "distance_km": distance,
    "weight_kg": weight,
    "transport_mode": transport_modes,
    "traffic_level": traffic_levels,
    "weather": weather_conditions,
    "planned_delivery_hours": planned_hours,
    "actual_delivery_hours": actual_hours,
    "transport_cost": transport_cost,
})
shipments["on_time"] = shipments["actual_delivery_hours"] <= shipments["planned_delivery_hours"]
shipments["delay_hours"] = (shipments["actual_delivery_hours"] - shipments["planned_delivery_hours"]).round(2)
shipments["cost_per_km"] = (shipments["transport_cost"] / shipments["distance_km"].clip(lower=1)).round(2)

# 3. Save initial data for use in later internship weeks
csv_path = OUTPUT_DIR / "week1_simulated_shipments.csv"
shipments.to_csv(csv_path, index=False)

# 4. Define and calculate preliminary KPIs
kpis = pd.DataFrame([
    {"KPI": "Total shipments", "Definition": "Number of shipment records", "Initial value": int(len(shipments))},
    {"KPI": "On-time delivery rate (%)", "Definition": "Shipments delivered on/before planned time divided by all shipments × 100", "Initial value": round(shipments["on_time"].mean() * 100, 2)},
    {"KPI": "Average actual delivery time (hours)", "Definition": "Mean actual delivery duration", "Initial value": round(shipments["actual_delivery_hours"].mean(), 2)},
    {"KPI": "Average delay (hours)", "Definition": "Mean actual delivery time minus planned delivery time; negative values mean early delivery", "Initial value": round(shipments["delay_hours"].mean(), 2)},
    {"KPI": "Average transport cost", "Definition": "Mean transport cost per shipment", "Initial value": round(shipments["transport_cost"].mean(), 2)},
    {"KPI": "Average cost per kilometre", "Definition": "Mean of shipment transport cost divided by distance", "Initial value": round(shipments["cost_per_km"].mean(), 2)},
])
kpis.to_csv(OUTPUT_DIR / "week1_kpi_definitions.csv", index=False)

# 5. Create a data dictionary
rows = [
    ("shipment_id", "Unique shipment identifier", "Text"),
    ("shipment_date", "Date associated with the shipment", "Date"),
    ("distance_km", "Shipment distance in kilometres", "Numeric"),
    ("weight_kg", "Shipment weight in kilograms", "Numeric"),
    ("transport_mode", "Mode used: Road, Rail, or Air", "Category"),
    ("traffic_level", "Illustrative traffic level: Low, Medium, or High", "Category"),
    ("weather", "Illustrative weather condition", "Category"),
    ("planned_delivery_hours", "Planned delivery duration in hours", "Numeric"),
    ("actual_delivery_hours", "Simulated actual delivery duration in hours", "Numeric"),
    ("transport_cost", "Illustrative transport cost in unspecified currency units", "Numeric"),
    ("on_time", "True if actual duration is less than or equal to planned duration", "Boolean"),
    ("delay_hours", "Actual duration minus planned duration; negative means early", "Numeric"),
    ("cost_per_km", "Transport cost divided by distance", "Numeric"),
]
pd.DataFrame(rows, columns=["Column", "Description", "Data type"]).to_csv(OUTPUT_DIR / "week1_data_dictionary.csv", index=False)

# 6. Save planning notes
planning_text = f"""WEEK 1 — STRATEGIC PLANNING AND DATA EXPLORATION IN LOGISTICS

Scenario:
{SCENARIO}

Business objective:
{BUSINESS_OBJECTIVE}

Business questions:
""" + "\n".join(f"{i}. {question}" for i, question in enumerate(BUSINESS_QUESTIONS, start=1)) + """

Proposed workflow:
1. Confirm the business problem and success measures.
2. Identify shipment-level fields and define each field clearly.
3. Review the sample data for structure, types, missing values, duplicates, and unusual values.
4. Clean and validate the dataset in Week 2.
5. Explore patterns and relationships in Week 3.
6. Build and evaluate a basic delivery-time prediction model in Week 4.

Data limitation:
All rows are simulated for educational purposes. KPI values are demonstration results,
not claims about a real logistics company. Currency units for transport_cost are unspecified.
"""
(OUTPUT_DIR / "week1_planning_summary.txt").write_text(planning_text, encoding="utf-8")

# 7. Display the initial exploration results
print("=" * 68)
print("WEEK 1: STRATEGIC PLANNING AND DATA EXPLORATION")
print("=" * 68)
print("Scenario:", SCENARIO)
print("Business objective:", BUSINESS_OBJECTIVE)
print("\nSample preview:")
print(shipments.head(5).to_string(index=False))
print("\nDataset shape (rows, columns):", shipments.shape)
print("\nColumn data types:")
print(shipments.dtypes.to_string())
print("\nMissing values by column:")
print(shipments.isna().sum().to_string())
print("\nPreliminary KPIs:")
print(kpis[["KPI", "Initial value"]].to_string(index=False))
print("\nSaved Week 1 files in:", OUTPUT_DIR.resolve())
print("- week1_simulated_shipments.csv")
print("- week1_kpi_definitions.csv")
print("- week1_data_dictionary.csv")
print("- week1_planning_summary.txt")
print("\nReminder: dataset and KPI results are simulated/illustrative only.")
