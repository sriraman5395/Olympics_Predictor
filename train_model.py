import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


athletes_df = pd.read_csv("athletes.csv")
noc_df = pd.read_csv("noc.csv")


df = athletes_df.merge(noc_df[['NOC', 'region']], how='left', on='NOC')


df = df[["Sport", "Sex", "Age", "Height", "Weight", "region", "Medal"]].dropna()


le_sport = LabelEncoder()
le_sex = LabelEncoder()
le_region = LabelEncoder()
le_medal = LabelEncoder()

df["Sport"] = le_sport.fit_transform(df["Sport"])
df["Sex"] = le_sex.fit_transform(df["Sex"])
df["region"] = le_region.fit_transform(df["region"])
df["Medal"] = le_medal.fit_transform(df["Medal"])


X = df.drop("region", axis=1)
y = df["region"]


label_counts = y.value_counts()
valid_labels = label_counts[label_counts >= 2].index
mask = y.isin(valid_labels)
X = X[mask].astype("float32") 
y = y[mask]


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)


with open("model.pkl", "wb") as f:
    joblib.dump(model, f, compress=3)


with open("label_encoders.pkl", "wb") as f:
    joblib.dump({
        "sport": le_sport,
        "sex": le_sex,
        "region": le_region,
        "medal": le_medal
    }, f)

print("✅ Model trained and saved as compressed model.pkl")
