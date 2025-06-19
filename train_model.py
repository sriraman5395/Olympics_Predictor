import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load datasets
athletes_df = pd.read_csv("athletes.csv")
noc_df = pd.read_csv("noc.csv")

# Merge on 'NOC' to get region
df = athletes_df.merge(noc_df[['NOC', 'region']], how='left', on='NOC')

# Drop rows with missing important values
df = df[["Sport", "Sex", "Age", "Height", "Weight", "region", "Medal"]].dropna()

# Encode categorical variables
le_sport = LabelEncoder()
le_sex = LabelEncoder()
le_region = LabelEncoder()
le_medal = LabelEncoder()

df["Sport"] = le_sport.fit_transform(df["Sport"])
df["Sex"] = le_sex.fit_transform(df["Sex"])
df["region"] = le_region.fit_transform(df["region"])
df["Medal"] = le_medal.fit_transform(df["Medal"])

# Features and target
X = df.drop("region", axis=1)
y = df["region"]

# Remove classes with <2 samples
label_counts = y.value_counts()
valid_labels = label_counts[label_counts >= 2].index
mask = y.isin(valid_labels)
X = X[mask].astype("float32")  # 👈 Convert to float32
y = y[mask]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train a smaller model
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save compressed model
with open("model.pkl", "wb") as f:
    joblib.dump(model, f, compress=3)

# Save the encoders
with open("label_encoders.pkl", "wb") as f:
    joblib.dump({
        "sport": le_sport,
        "sex": le_sex,
        "region": le_region,
        "medal": le_medal
    }, f)

print("✅ Model trained and saved as compressed model.pkl")
