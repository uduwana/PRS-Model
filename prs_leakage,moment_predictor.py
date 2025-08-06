import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load CFD dataset
data = pd.read_excel("cfd_data.xlsx", sheet_name="Simulation_Summary")

# Select inputs
X = data[[
    'No. of Pockets', 'Pocket Angle (°)', 'Pocket Depth (mm)',
    'Pocket Width (mm)', 'Inner Radius (mm)', 'Outer Radius (mm)',
    'Film Thickness (μm)', 'Inlet Pressure (MPa)', 'RPM',
    'Tilt Angle (°)', 'Viscosity (Pa·s)', 'Density (kg/m³)'
]]

# Select multiple outputs
y_leakage = data['Leakage Rate (L/min)']
y_moment = data['Restoring Moment (Nm)']

# Polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Train PRS model for Leakage Rate
model_leakage = LinearRegression()
model_leakage.fit(X_poly, y_leakage)

# Train PRS model for Restoring Moment
model_moment = LinearRegression()
model_moment.fit(X_poly, y_moment)

# Evaluate both models
y_pred_leakage = model_leakage.predict(X_poly)
y_pred_moment = model_moment.predict(X_poly)

mae_leakage = mean_absolute_error(y_leakage, y_pred_leakage)
r2_leakage = r2_score(y_leakage, y_pred_leakage)

mae_moment = mean_absolute_error(y_moment, y_pred_moment)
r2_moment = r2_score(y_moment, y_pred_moment)

print(f"\n Model Evaluation on Training Data:")
print(f" -Leakage Rate  - MAE: {mae_leakage:.4f}, R²: {r2_leakage:.4f}")
print(f" -Restoring Moment - MAE: {mae_moment:.4f}, R²: {r2_moment:.4f}")

# New design input
new_input_values = [[
    3,      # No. of Pockets
    10,     # Pocket Angle (°)
    2.0,    # Pocket Depth (mm)
    5.0,    # Pocket Width (mm)
    12,     # Inner Radius (mm)
    20,     # Outer Radius (mm)
    35,     # Film Thickness (μm)
    2.0,    # Inlet Pressure (MPa)
    1500,   # RPM
    10,     # Tilt Angle (°)
    0.0012, # Viscosity (Pa·s)
    880     # Density (kg/m³)
]]
new_input_df = pd.DataFrame(new_input_values, columns=X.columns)
new_input_poly = poly.transform(new_input_df)

# Predict for new input
predicted_leakage = model_leakage.predict(new_input_poly)
predicted_moment = model_moment.predict(new_input_poly)

print(f"\n Predictions for New Input:")
print(f" -Predicted Leakage Rate: {predicted_leakage[0]:.4f} L/min")
print(f" -Predicted Restoring Moment: {predicted_moment[0]:.4f} Nm")

# Plot actual vs predicted for Leakage Rate
plt.figure(figsize=(6, 5))
plt.scatter(y_leakage, y_pred_leakage, color='blue', label='Leakage Rate')
plt.plot([y_leakage.min(), y_leakage.max()],
         [y_leakage.min(), y_leakage.max()],
         'r--', label='Ideal Fit')
plt.xlabel("Actual Leakage Rate (L/min)")
plt.ylabel("Predicted Leakage Rate (L/min)")
plt.title("PRS Model: Leakage Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
