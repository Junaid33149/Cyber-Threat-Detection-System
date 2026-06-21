import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

from src.utils import preprocess_data

# ==============================
# CREATE FOLDERS
# ==============================
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ==============================
# LOAD & MERGE DATA (FIXED)
# ==============================
print("📥 Loading dataset...")

files = glob.glob("data/*.csv")

if len(files) == 0:
    print("❌ No CSV files found in data/ folder")
    exit()

df_list = []

for file in files:
    try:
        print(f"📂 Loading: {file}")
        df = pd.read_csv(file)
        df_list.append(df)
    except Exception as e:
        print(f"⚠️ Skipping {file}: {e}")

# Combine all CSVs
data = pd.concat(df_list, ignore_index=True)

# Check empty
if data.empty:
    print("❌ Dataset is empty after merging")
    exit()

print("✅ All files merged successfully")
print("Shape:", data.shape)
print(data.head())

# ==============================
# 🚀 SPEED OPTIMIZATION
# ==============================
print("⚡ Reducing dataset size for faster training...")

if len(data) > 50000:
    data = data.sample(n=50000, random_state=42)

print("New Shape:", data.shape)

# ==============================
# PREPROCESS
# ==============================
X, y, scaler, feature_names = preprocess_data(data)

print("✅ Data Preprocessed")

# ==============================
# SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# TRAIN MODEL
# ==============================
print("🤖 Training model...")

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

print("✅ Model Trained")

# ==============================
# PREDICT
# ==============================
y_pred = model.predict(X_test)

# ==============================
# EVALUATE
# ==============================
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")

print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ==============================
# CONFUSION MATRIX (CLEAN)
# ==============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

plt.title("Confusion Matrix", fontsize=15, weight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=300)
plt.close()

# ==============================
# FEATURE IMPORTANCE (CLEAN)
# ==============================
importances = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)

# Shorten long names
feat_df["Feature"] = feat_df["Feature"].apply(lambda x: x[:25])

plt.figure(figsize=(10,6))
sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")

plt.title("Top 10 Important Features", fontsize=15, weight='bold')
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=300)
plt.close()

# ==============================
# CLASS DISTRIBUTION (CLEAN)
# ==============================
plt.figure(figsize=(6,5))
sns.countplot(x=y, palette="Set2")

plt.title("Attack vs Normal Distribution", fontsize=15, weight='bold')
plt.xlabel("Class (0 = Normal, 1 = Attack)")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("images/class_distribution.png", dpi=300)
plt.close()

# ==============================
# CORRELATION HEATMAP (CLEAN)
# ==============================
plt.figure(figsize=(12,10))

numeric_data = data.select_dtypes(include=['number'])

if not numeric_data.empty:
    
    corr = numeric_data.corr().abs()

    # pick top 15 important features
    top_features = corr.sum().sort_values(ascending=False).head(15).index

    corr_subset = numeric_data[top_features].corr()

    sns.heatmap(
        corr_subset,
        cmap='coolwarm',
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Correlation Heatmap (Top 15 Features)", fontsize=15, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig("images/correlation_heatmap.png", dpi=300)
    plt.close()
else:
    print("⚠️ No numeric columns for correlation heatmap")

print("📈 All graphs saved in images/")

# ==============================
# SAVE MODEL
# ==============================
joblib.dump(model, "models/cyber_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("💾 Model saved successfully")

# ==============================
# SAMPLE TEST
# ==============================
sample = X_test[0].reshape(1, -1)
pred = model.predict(sample)

print("\n🔍 Sample Prediction:", pred)