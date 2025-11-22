import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_tabular

# Load the dataset
data = pd.read_csv("phishing_with_label.csv")  # Using the available dataset in your workspace

# Prepare features and target
X = data.drop(columns=['Label'])  # Features
y = data['Label']                 # Target

# Get feature names for LIME
feature_names = list(X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

print(f"Best KNN parameters: {grid.best_params_}")
print(f"Best cross-validated accuracy: {grid.best_score_:.4f}")

# Use best estimator
knn = grid.best_estimator_
knn.fit(X_train_scaled, y_train)

# Evaluate
y_pred = knn.predict(X_test_scaled)
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create and plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('knn_confusion_matrix.png')
plt.close()

# Initialize LIME explainer
print("\nInitializing LIME explainer...")
explainer = lime_tabular.LimeTabularExplainer(
    X_train_scaled,
    feature_names=feature_names,
    class_names=['Legitimate', 'Phishing'],
    mode='classification',
    training_labels=y_train
)

# Function to explain predictions
def explain_prediction(instance_idx, num_features=10):
    print(f"\nExplaining prediction for instance {instance_idx}...")
    
    # Get the instance
    instance = X_test_scaled[instance_idx]
    true_label = y_test.iloc[instance_idx]
    
    # Generate explanation
    exp = explainer.explain_instance(
        instance, 
        knn.predict_proba,
        num_features=num_features
    )
    
    # Get prediction probabilities
    probs = knn.predict_proba([instance])[0]
    predicted_label = "Phishing" if probs[1] > 0.5 else "Legitimate"
    
    print(f"True label: {true_label}")
    print(f"Predicted label: {predicted_label}")
    print(f"Prediction probabilities: Legitimate: {probs[0]:.2f}, Phishing: {probs[1]:.2f}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.title(f'LIME Explanation (Instance {instance_idx})\nPredicted: {predicted_label} ({max(probs):.2%} confidence)')
    plt.tight_layout()
    plt.savefig(f'lime_explanation_{instance_idx}.png')
    plt.close()
    
    # Print feature importance
    print("\nFeature Importance:")
    for feature, importance in exp.as_list():
        print(f"{feature}: {importance:.3f}")

# Explain multiple predictions
print("\nGenerating explanations for sample instances...")
for idx in [0, 1, 2]:  # Explain first 3 test instances
    explain_prediction(idx)

# Create summary visualization of feature importance across instances
def visualize_feature_importance_summary(num_instances=10, top_features=10):
    print(f"\nAnalyzing feature importance across {num_instances} instances...")
    all_explanations = []
    
    # Get explanations for multiple instances
    for idx in range(min(num_instances, len(X_test_scaled))):
        exp = explainer.explain_instance(
            X_test_scaled[idx],
            knn.predict_proba,
            num_features=top_features
        )
        all_explanations.append(dict(exp.as_list()))
    
    # Calculate average absolute importance
    feature_importance = {}
    for exp in all_explanations:
        for feature, importance in exp.items():
            if feature not in feature_importance:
                feature_importance[feature] = []
            feature_importance[feature].append(abs(importance))
    
    # Calculate mean importance
    mean_importance = {
        feature: np.mean(values)
        for feature, values in feature_importance.items()
    }
    
    # Plot top features
    plt.figure(figsize=(12, 6))
    features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)[:top_features]
    features, importances = zip(*features)
    
    plt.barh(features, importances)
    plt.title(f'Average Feature Importance Across {num_instances} Instances')
    plt.xlabel('Average Absolute Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_summary.png')
    plt.close()

# Generate feature importance summary
visualize_feature_importance_summary()
