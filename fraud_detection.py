import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load dataset
df = pd.read_csv('fraudTest.csv')

# Drop high-cardinality or identifier columns
df = df.drop(['trans_date_trans_time', 'dob', 'unix_time', 'merchant', 'category',
              'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'cc_num'], axis=1)

# If there are still low-cardinality categorical features, encode them
# Here we assume 'trans_num' is safe; if it's unique per row, drop that too
df = df.drop(['trans_num'], axis=1)  # safer if it's unique per transaction

# Define X and y
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
