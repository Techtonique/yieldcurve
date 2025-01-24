import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from yieldcurveml.interpolatecurve import CurveInterpolator

# Your data (divided by 100)
yM = np.asarray([9.193782, 9.502359, 9.804080, 9.959691, 10.010291, 
      9.672974, 9.085818, 8.553107, 8.131273, 7.808959, 
      7.562701, 7.371855, 7.221084, 7.099587])/100

tm = np.asarray([0.08333333, 0.25000000, 0.50000000, 0.75000000, 1.00000000, 
      2.00000000, 3.00000000, 4.00000000, 5.00000000, 6.00000000, 
      7.00000000, 8.00000000, 9.00000000, 10.00000000])

# Define models to compare
models = {
    'Extra Trees': ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=1, min_samples_split=2),
    'Ridge': RidgeCV(alphas=10**np.linspace(-10, 10, 100)),
    'KRR': KernelRidge(alpha=0.1, kernel='rbf'),
    'Linear': LinearRegression()
}

# Create subplot figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot for Laguerre basis
ax1.scatter(tm, yM, color='black', label='Original points', zorder=5)
for name, model in models.items():
    interpolator = CurveInterpolator(estimator=model, type_regressors="laguerre")
    interpolator.fit(tm, yM)
    yM_interp = interpolator.predict(tm)
    ax1.plot(tm, yM_interp.spot_rates, label=f'{name}', alpha=0.7)

ax1.set_xlabel('Maturity (years)')
ax1.set_ylabel('Yield')
ax1.set_title('Yield Curve Interpolation - Laguerre Basis')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot for Cubic basis
ax2.scatter(tm, yM, color='black', label='Original points', zorder=5)
for name, model in models.items():
    interpolator = CurveInterpolator(estimator=model, type_regressors="cubic")
    interpolator.fit(tm, yM)
    yM_interp = interpolator.predict(tm)
    ax2.plot(tm, yM_interp.spot_rates, label=f'{name}', alpha=0.7)

ax2.set_xlabel('Maturity (years)')
ax2.set_ylabel('Yield')
ax2.set_title('Yield Curve Interpolation - Cubic Basis')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print residuals for each model and basis
print("\nResiduals (RMSE):")
for basis in ["laguerre", "cubic"]:
    print(f"\n{basis.capitalize()} basis:")
    for name, model in models.items():
        interpolator = CurveInterpolator(estimator=model, type_regressors=basis)
        interpolator.fit(tm, yM)
        yM_interp = interpolator.predict(tm)
        rmse = np.sqrt(np.mean((yM - yM_interp.spot_rates) ** 2))
        print(f"{name}: {rmse:.6f}")