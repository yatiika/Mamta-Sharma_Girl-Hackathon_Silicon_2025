from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset (assuming we have historical RTL feature data)
df = pd.read_csv("rtl_feature_dataset.csv")

X = df.drop(columns=["actual_combinational_depth"])
y = df["actual_combinational_depth"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))

# Save trained model
joblib.dump(model, "depth_predictor.pkl")
