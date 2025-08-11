import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def target_encode_categorical_features(
    df:pd.DataFrame, 
    categorical_columns:list[str], 
    target_column: str
) -> pd.DataFrame:
    """Codifica cada variável categórica de uma lista, conforme a média do target para cada categoria"""
    
    # Duplica o dataset que será retornado com as colunas codificadas
    encoded_data = df.copy()
    
    # Itera cada coluna da lista
    for col in categorical_columns:
        
        # Constrói um mapa de codificação com a média por categoria da coluna
        encoding_map = df.groupby(col)[target_column].mean().to_dict()
        
        # Aplica a o mapa de codificação para substituir a respectiva coluna
        encoded_data[col] = encoded_data[col].map(encoding_map)
    
    return (encoded_data)

def impute_and_scale_data(
    df_features:pd.DataFrame
) -> pd.DataFrame:
    
    # Fill missing values with SimpleImputer using mean as strategy
    imputer = SimpleImputer(strategy="mean")
    X_preprocessed = imputer.fit_transform(df_features.values)
    
    # Scale data
    scaler = StandardScaler()
    X_preprocessed = scaler.fit_transform(X_preprocessed)
    
    return pd.DataFrame(X_preprocessed,columns=df_features.columns)

# Load file
data = pd.read_csv("./data/weather.csv")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(TARGET_COLUMN), 
    data[TARGET_COLUMN], 
    random_state=1993
)

# Instantiate the classifier
clf = RandomForestClassifier(
    max_depth=2,
    n_estimators=50,
    random_state=1993
)

# Train
clf.fit(X_train,y_train)

# Predict
y_pred = clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Plot Confusion Matrix
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap=plt.cm.Blues)
