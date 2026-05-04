from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("customer_data.csv")

# Use ALL features
x = df[["Age", "Income", "Education", "Experience"]]
y = df["Bought"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(x, y)
model.fit(x_train, y_train)

# Predict on test set
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# predict on new data
new_data = pd.DataFrame([[25,50000,1,2]], columns=["Age", "Income", "Education", "Experience"])
print(model.predict(new_data))

# Plot tree
plt.figure(figsize=(12,5))
plot_tree(
    model,
    feature_names=x.columns,
    class_names=["No Purchase", "Purchase"],
    filled=True
)

plt.show()