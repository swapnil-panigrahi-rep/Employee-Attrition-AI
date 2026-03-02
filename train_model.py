import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ===============================
# LOAD DATASET
# ===============================

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("Dataset Loaded")
print("Shape:", df.shape)


# ===============================
# SELECT ONLY REQUIRED COLUMNS
# ===============================

df = df[[
'Age',
'MonthlyIncome',
'JobRole',
'OverTime',
'BusinessTravel',
'JobSatisfaction',
'YearsAtCompany',
'Attrition'
]]


# ===============================
# SPLIT FEATURES & TARGET
# ===============================

X = df.drop('Attrition',axis=1)
y = df['Attrition']


# Encode Target

le = LabelEncoder()
y = le.fit_transform(y)


# ===============================
# ONE HOT ENCODING
# ===============================

X = pd.get_dummies(X)


# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train,X_test,y_train,y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)


# ===============================
# SCALING
# ===============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ===============================
# MODEL TRAINING
# ===============================

model = RandomForestClassifier()

model.fit(X_train,y_train)


# ===============================
# ACCURACY
# ===============================

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))


# ===============================
# SAVE FILES
# ===============================

joblib.dump(model,"attrition_model.pkl")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(X.columns,"columns.pkl")

print("Model Saved")