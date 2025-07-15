import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your CSV file
df = pd.read_csv('creditcard.csv')  # Replace with your CSV file name

# Assume the last column is the target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a model with verbose output
model = RandomForestClassifier(random_state=42, verbose=1)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")