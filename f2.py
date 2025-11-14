import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: Load PhishTank dataset
# ----------------------------
df = pd.read_csv("phishtank_data.csv")
print("PhishTank dataset loaded:", df.shape)

# PhishTank contains only PHISHING URLs
df["label"] = 1
df = df[["url", "label"]]

# ----------------------------
# Step 2: Create legitimate URLs
# ----------------------------
legit_urls = [
    "https://www.google.com/",
    "https://github.com/",
    "https://www.wikipedia.org/",
    "https://www.reddit.com/",
    "https://www.yahoo.com/",
    "https://www.microsoft.com/",
    "https://www.stackoverflow.com/",
    "https://www.python.org/",
    "https://www.kaggle.com/",
    "https://www.coursera.org/"
] * 500  # repeat to balance dataset

legit_df = pd.DataFrame({"url": legit_urls, "label": 0})

# ----------------------------
# Step 3: Combine phishing + legit
# ----------------------------
full_df = pd.concat([df, legit_df], ignore_index=True)
print("Combined dataset:", full_df.shape)

# ----------------------------
# Step 4: Lightweight URL feature extraction
# ----------------------------
def extract_features(url):
    return {
        "url_length": len(url),
        "num_dots": url.count('.'),
        "num_slash": url.count('/'),
        "num_hyphen": url.count('-'),
        "has_https": 1 if url.startswith("https://") else 0,
        "digit_ratio": sum(c.isdigit() for c in url) / len(url),
        "alpha_ratio": sum(c.isalpha() for c in url) / len(url),
    }

features = full_df["url"].apply(extract_features).apply(pd.Series)
features["label"] = full_df["label"]

# ----------------------------
# Step 5: Train-test split
# ----------------------------
X = features.drop("label", axis=1)
y = features["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ----------------------------
# Step 6: Train a Decision Tree
# ----------------------------
model = DecisionTreeClassifier(
    max_depth=8,             # keep the tree small
    min_samples_split=5,
    class_weight="balanced", # handle imbalance
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Step 7: Evaluation
# ----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# Step 8: Plot Decision Tree
# ----------------------------
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Legit", "Phish"],
    filled=True,
    fontsize=8
)
plt.title("Decision Tree for Phishing Detection")
plt.show()
