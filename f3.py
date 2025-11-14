import pandas as pd
import numpy as np
import re
import random
import tldextract
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# -------------------------------------------------------
# LIGHTWEIGHT URL FEATURE EXTRACTOR (10 FEATURES)
# -------------------------------------------------------

def extract_light_features(url):
    parsed = urlparse(url)
    domain_info = tldextract.extract(url)

    f = {}
    f['url_length'] = len(url)
    f['num_dots'] = url.count('.')
    f['num_hyphens'] = url.count('-')
    f['num_digits'] = sum(c.isdigit() for c in url)
    f['num_subdomains'] = len(domain_info.subdomain.split('.')) if domain_info.subdomain else 0
    f['tld_length'] = len(domain_info.suffix)

    keywords = ["login", "secure", "verify", "update", "bank", "account"]
    f['suspicious_keyword_count'] = sum(1 for w in keywords if w in url.lower())

    f['has_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0

    f['digit_ratio'] = f['num_digits'] / len(url)
    f['letter_ratio'] = sum(c.isalpha() for c in url) / len(url)

    return f


# -------------------------------------------------------
# LOAD PHISHTANK DATA (PHISHING URLs)
# -------------------------------------------------------
df_phish = pd.read_csv("phishtank_data.csv")
df_phish = df_phish[['url']].dropna()
df_phish['label'] = 1

print("Phishing URLs loaded:", len(df_phish))


# -------------------------------------------------------
# GENERATE LEGITIMATE URL DATASET
# -------------------------------------------------------

def generate_legit_url():
    domains = [
        "google.com", "github.com", "stackoverflow.com",
        "numpy.org", "scikit-learn.org", "python.org",
        "kaggle.com", "wikipedia.org"
    ]
    paths = ["docs", "learn", "guide", "help", "topics", "examples"]

    return "https://" + random.choice(domains) + "/" + random.choice(paths)

df_legit = pd.DataFrame({
    "url": [generate_legit_url() for _ in range(len(df_phish))],
    "label": 0
})

print("Legitimate URLs generated:", len(df_legit))


# -------------------------------------------------------
# COMBINE DATASETS
# -------------------------------------------------------
df = pd.concat([df_phish, df_legit], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Final dataset size:", len(df))


# -------------------------------------------------------
# EXTRACT LIGHTWEIGHT FEATURES
# -------------------------------------------------------
features = [extract_light_features(url) for url in df['url']]
X = pd.DataFrame(features)
y = df['label']

print("Feature matrix shape:", X.shape)


# -------------------------------------------------------
# TRAIN-TEST SPLIT
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------
# LIGHTWEIGHT MODEL (LOGISTIC REGRESSION)
# -------------------------------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Lightweight Model Accuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -------------------------------------------------------
# FEATURE IMPORTANCE PLOT
# -------------------------------------------------------
importance = np.abs(model.coef_[0])  # absolute weights
feature_names = X.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
plt.title("Lightweight Model Feature Importance")
plt.xlabel("Importance (Absolute Coefficient)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
