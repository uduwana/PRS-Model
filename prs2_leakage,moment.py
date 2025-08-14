import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# 1) Load data
data = pd.read_excel("cfd_data.xlsx", sheet_name="Simulation_Summary")

# Inputs (as in your Excel)
X = data[[
    'No. of Pockets', 'Pocket Angle (°)', 'Pocket Depth (mm)',
    'Pocket Width (mm)', 'Inner Radius (mm)', 'Outer Radius (mm)',
    'Film Thickness (μm)', 'Inlet Pressure (MPa)', 'RPM',
    'Tilt Angle (°)', 'Viscosity (Pa·s)', 'Density (kg/m³)'
]].copy()

# Outputs
y_leakage = data['Leakage Rate (L/min)'].values
y_moment  = data['Restoring Moment (Nm)'].values

# 2) Code inputs to [-1, +1]  (RSM practice)
#    x_coded = 2*(x - min)/(max-min) - 1
#    Save mins/ranges to code new designs the same way.

mins = X.min()
ranges = (X.max() - X.min()).replace(0, 1.0)  # avoid zero-division
X_coded = 2*(X - mins)/ranges - 1

# Feature names for coded variables (used to print PRS equation)
coded_names = [f"C_{c}" for c in X.columns]


# 3) Build full quadratic features (degree=2 = linear + interactions + squares)
poly = PolynomialFeatures(degree=2, include_bias=False)
# Fit on coded variables
X_poly = poly.fit_transform(X_coded.values)
feature_names = poly.get_feature_names_out(coded_names)  # names of terms in PRS


# 4) Fit PRS models (least squares) for both outputs
model_leak = LinearRegression().fit(X_poly, y_leakage)
model_mom  = LinearRegression().fit(X_poly, y_moment)


# 5) Training-fit evaluation (on available data)
yhat_leak = model_leak.predict(X_poly)
yhat_mom  = model_mom.predict(X_poly)

print("\n📊 Model Evaluation on Training Data (coded PRS, degree=2)")
print(f" - Leakage Rate     MAE: {mean_absolute_error(y_leakage, yhat_leak):.6f}, R²: {r2_score(y_leakage, yhat_leak):.6f}")
print(f" - Restoring Moment MAE: {mean_absolute_error(y_moment,  yhat_mom ): .6f}, R²: {r2_score(y_moment,  yhat_mom ): .6f}")


# 6) Define a NEW design (real units), code it, and predict
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
new_df_real = pd.DataFrame(new_input_values, columns=X.columns)

# Code the new point with SAME mins/ranges
new_df_coded = 2*(new_df_real - mins)/ranges - 1
new_poly     = poly.transform(new_df_coded.values)

pred_leak = model_leak.predict(new_poly)[0]
pred_mom  = model_mom.predict(new_poly)[0]


# 7) Print the new inputs line-by-line and predictions
print("\n Predictions for New Input (real units)")
print("Given input variables:")
for col, val in zip(new_df_real.columns, new_input_values[0]):
    print(f" - {col}: {val}")

print(f"\n - Predicted Leakage Rate:     {pred_leak:.6f} L/min")
print(f" - Predicted Restoring Moment: {pred_mom:.6f} Nm")


# 8) Print the explicit PRS equations (in CODED variables)
#    y = b0 + sum(b_i * C_xi) + sum(b_ij * C_xi*C_xj) + sum(b_ii * C_xi^2)

def print_prs_equation(model, feat_names, title):
    intercept = model.intercept_
    coefs = model.coef_
    terms = [f"{intercept:.6g}"] + [f"{coefs[i]:.6g}*{feat_names[i]}" for i in range(len(feat_names))]
    eq = " + ".join(terms).replace("+ -", "- ")
    print(f"\n PRS equation (coded vars) — {title}:\n{title} = {eq}\n")

print_prs_equation(model_leak, feature_names, "Leakage")
print_prs_equation(model_mom,  feature_names, "RestoringMoment")


# 9) Plot (Actual vs Predicted) for Leakage on training data
plt.figure(figsize=(6, 5))
plt.scatter(y_leakage, yhat_leak, color='blue', label='Leakage (train fit)')
mn, mx = y_leakage.min(), y_leakage.max()
plt.plot([mn, mx], [mn, mx], 'r--', label='Ideal Fit')
plt.xlabel("Actual Leakage Rate (L/min)")
plt.ylabel("Predicted Leakage Rate (L/min)")
plt.title("PRS (coded, degree=2): Leakage — Training Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
