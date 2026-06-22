import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# SETTINGS
# =====================================================

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

plt.ion()

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("weather.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# =====================================================
# MISSING VALUES
# =====================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.savefig("missing_values_heatmap.png")
plt.close()

# Fill missing values

for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# =====================================================
# RAINFALL DISTRIBUTION
# =====================================================

plt.figure(figsize=(8,5))
sns.histplot(df["Rainfall"], bins=30, kde=True)
plt.title("Rainfall Distribution")
plt.savefig("rainfall_distribution.png")
plt.close()

# =====================================================
# OUTLIER DETECTION
# =====================================================

plt.figure(figsize=(12,6))
sns.boxplot(data=df[["MinTemp","MaxTemp","Rainfall"]])
plt.title("Outlier Detection")
plt.savefig("outlier_detection.png")
plt.close()

# =====================================================
# CORRELATION HEATMAP
# =====================================================

plt.figure(figsize=(14,10))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.close()

# =====================================================
# PAIRPLOT
# =====================================================

sample_size = min(300, len(df))

sns.pairplot(
    df[["MinTemp","MaxTemp","Rainfall"]].sample(sample_size)
)

plt.savefig("pairplot.png")
plt.close()

# =====================================================
# RAIN TODAY ANALYSIS
# =====================================================

plt.figure(figsize=(6,4))
sns.countplot(x="RainToday", data=df)
plt.title("Rain Today Distribution")
plt.savefig("rain_today_distribution.png")
plt.close()

# =====================================================
# RAIN TOMORROW ANALYSIS
# =====================================================

plt.figure(figsize=(6,4))
sns.countplot(x="RainTomorrow", data=df)
plt.title("Rain Tomorrow Distribution")
plt.savefig("rain_tomorrow_distribution.png")
plt.close()

# =====================================================
# HUMIDITY VS RAINFALL
# =====================================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="Humidity3pm",
    y="Rainfall",
    data=df
)

plt.title("Humidity vs Rainfall")
plt.savefig("humidity_vs_rainfall.png")
plt.close()

# =====================================================
# CORRELATION WITH RAINFALL
# =====================================================

corr_rain = df.corr(numeric_only=True)["Rainfall"]

print("\n========== CORRELATION WITH RAINFALL ==========")
print(corr_rain.sort_values(ascending=False))

# =====================================================
# MACHINE LEARNING
# =====================================================

features = [
    "MinTemp",
    "MaxTemp",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Temp9am",
    "Temp3pm",
    "WindSpeed9am",
    "WindSpeed3pm"
]

X = df[features]
y = df["Rainfall"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# MODEL COMPARISON
# =====================================================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []

best_model = None
best_r2 = -999

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append(
        [name, mae, mse, rmse, r2]
    )

    print(f"\n{name}")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_predictions = predictions

# =====================================================
# MODEL RESULTS
# =====================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "MSE",
        "RMSE",
        "R2 Score"
    ]
)

print("\n========== MODEL PERFORMANCE ==========")
print(results_df)

results_df.to_csv(
    "model_performance.csv",
    index=False
)

# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    best_predictions
)

plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("Actual vs Predicted Rainfall")

plt.savefig("actual_vs_predicted.png")
plt.close()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })

    print("\n========== FEATURE IMPORTANCE ==========")
    print(
        importance.sort_values(
            by="Importance",
            ascending=False
        )
    )

    plt.figure(figsize=(10,6))

    sns.barplot(
        data=importance.sort_values(
            by="Importance",
            ascending=False
        ),
        x="Importance",
        y="Feature"
    )

    plt.title("Feature Importance")

    plt.savefig("feature_importance.png")
    plt.close()

# =====================================================
# SAVE SUMMARY REPORT
# =====================================================

df.describe().to_csv(
    "weather_summary.csv"
)

with open(
    "analysis_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("WEATHER ANALYSIS REPORT\n")
    f.write("="*50)

    f.write("\n\nDataset Shape:\n")
    f.write(str(df.shape))

    f.write("\n\nColumns:\n")
    f.write(str(df.columns.tolist()))

    f.write("\n\nModel Performance:\n")
    f.write(str(results_df))

    f.write("\n\nBest Model:\n")
    f.write(type(best_model).__name__)

    f.write("\n\nBest R2 Score:\n")
    f.write(str(best_r2))

print("\n===================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("===================================")

print("\nGenerated Files:")
print("1. missing_values_heatmap.png")
print("2. rainfall_distribution.png")
print("3. outlier_detection.png")
print("4. correlation_matrix.png")
print("5. pairplot.png")
print("6. rain_today_distribution.png")
print("7. rain_tomorrow_distribution.png")
print("8. humidity_vs_rainfall.png")
print("9. actual_vs_predicted.png")
print("10. feature_importance.png")
print("11. weather_summary.csv")
print("12. model_performance.csv")
print("13. analysis_report.txt")
