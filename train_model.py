import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
 
# Step 1: Load dataset
df = pd.read_csv("sprint_data_sample_300_with_ids.csv.crdownload")
 
# Step 2: Clean column names (remove spaces)
df.columns = df.columns.str.strip()
 
print("columns in my csv:", df.columns.tolist())
 
# Step 2.1: Create Delay Category based on incomplete points
df['Incomplete Points'] = df['Planned Story Points'] - df['Completed Story Points']
 
def delay_category(row):
    if row['Delay'] == 'No':
        return 'No Delay'
    else:
        if row['Incomplete Points'] <= 10:
            return 'Short Delay'
        elif row['Incomplete Points'] <= 30:
            return 'Medium Delay'
        else:
            return 'Long Delay'
 
df['Delay Category'] = df.apply(delay_category, axis=1)
 
# Step 3: Encode categorical columns (except target)
df_encoded = df.copy()
label_encoders = {}
categorical_cols = ["Unavailability", "Previous Sprint Delay"]
 
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
 
# Encode Delay Category target
le_delay_cat = LabelEncoder()
df_encoded['Delay Category Encoded'] = le_delay_cat.fit_transform(df['Delay Category'])

# Step 4: Define features and target
features = [
    "Planned Story Points", "Completed Story Points", "Sprint Duration",
    "Blockers Count", "Team Size", "Holidays",
    "Unavailability", "Previous Sprint Delay"
]
target = "Delay Category Encoded"
 
X = df_encoded[features]
y = df_encoded[target]
 
# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Step 6: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
 
# Step 7: Evaluate the model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Report:\n", classification_report(y_test, y_pred, target_names=le_delay_cat.classes_))
 
# Step 8: Save the model and encoders
joblib.dump(model, "delay_category_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(le_delay_cat, "delay_category_label_encoder.pkl")