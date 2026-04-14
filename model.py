import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load dataset
df = pd.read_csv("event_data.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Print columns to check
print("Columns in dataset:")
print(df.columns)

# 🔥 Step 2: Print actual column names (for debugging)
print("Original Columns:")
print(df.columns)

# Step 3: Rename columns properly
df.columns = df.columns.str.replace('\n', '').str.strip()
df = df.rename(columns={
    "What is your department?": "Department",
    "Which year are you in?": "Year",
    "Which type of events are you interested in?": "Interest",
    "What is your preferred time for events?": "Time",
    "How often do you participate in college events?": "Frequency",
    "Which event would you most likely attend?": "Event"
})
# Step 4: Drop Timestamp column
if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])

print("\nRenamed Columns:")
print(df.columns)

print("\nData Preview:")
print(df.head())

# Step 5: Encode text → numbers
le = LabelEncoder()

for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Step 6: Split data
X = df[['Department', 'Year', 'Interest', 'Time', 'Frequency']]
y = df['Event']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 7: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 8: Accuracy
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Step 9: Save model
pickle.dump(model, open("model.pkl", "wb"))

print("\n Model trained and saved successfully!")