import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from metrics_and_plots import plot_confusion_matrix, save_metrics
import json 

def encode_target (target):
    encoded_target = [1 if i == "Yes" else 0 for i in target]
    return encoded_target
    
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

def evaluate_model(model, X_test, y_test, float_precision=4):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return json.loads(
        json.dumps(metrics), parse_float=lambda x: round(float(x), float_precision)
    )

# Load file
def load_data(file_path):
    data = pd.read_csv(file_path)
    #X = data.drop(TARGET_COLUMN, axis=1)
    #y = data[TARGET_COLUMN]
    #return X, y
    return data

def main():
    # X, y = load_data("./data/weather.csv")
    data = load_data("./data/weather.csv")
    data.drop("Unnamed: 0",axis=1,inplace=True)
    
    # Encode Target
    data[TARGET_COLUMN] = encode_target(data[TARGET_COLUMN].to_list())
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("int")
    
    # Encode categorical features
    categorical_columns = ["Location","WindGustDir","WindDir9am","WindDir3pm","RainToday"]
    data_encoded= target_encode_categorical_features(data, categorical_columns,TARGET_COLUMN)
    
    # Imput and Scale Data
    data_encoded.drop("Date",inplace=True,axis=1)
    data_scaled= impute_and_scale_data(data_encoded)
    data_scaled[TARGET_COLUMN] = data_scaled[TARGET_COLUMN].astype("int")

    X = data_scaled.drop(TARGET_COLUMN, axis=1)
    y = data_scaled[TARGET_COLUMN]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        random_state=1993
    )
    
    # Encode categorical features
    #categorical_columns = ["Location","WindGustDir","WindDir9am","WindDir3pm","RainToday"]
    #X_train_encoded= target_encode_categorical_features(X_train, categorical_columns,TARGET_COLUMN)
    #X_test_encoded= target_encode_categorical_features(X_test, categorical_columns,TARGET_COLUMN)

    # Imput and Scale Data
    #X_train_scaled= impute_and_scale_data(X_train_encoded)
    #X_test_scaled= impute_and_scale_data(X_test_encoded)
    
    # Instantiate the classifier
    clf = RandomForestClassifier(
        max_depth=2,
        n_estimators=50,
        random_state=1993
    )

    # Train
    clf.fit(X_train,y_train)

    metrics = evaluate_model(clf, X_test, y_test)

    print("====================Test Set Metrics==================")
    print(json.dumps(metrics, indent=2))
    print("======================================================")

    save_metrics(metrics)
    plot_confusion_matrix(clf, X_test, y_test)

TARGET_COLUMN="RainTomorrow"
if __name__ == "__main__":
    main()
