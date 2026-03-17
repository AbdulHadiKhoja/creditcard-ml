import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import joblib
df = pd.read_csv(r"C:\Users\Lenovo\Desktop\DataSets\creditcard.csv")

def printScores(val,train,test,pred,con_matrix,class_report):
    print(f"ValidationScore:{val}")
    print(f"TrainScore: {train}")
    print(f"TestScore: {test}")
    print(f"Predict: {pred}")
    print(f"ConfusionMatrix: {con_matrix}")
    print(f"Report: {class_report}")

X = df.drop("Class",axis=1)
y = df["Class"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=42)

model = joblib.load("model.pkl")

test_pred = model.predict(X_test)
val_pred = model.predict(X_val)

trainscore = model.score(X_train,y_train)
testscore = model.score(X_test,y_test)
valscore = model.score(X_val,y_val)
con_matrix = confusion_matrix(y_test,test_pred)
class_report = classification_report(y_test,test_pred)
printScores(valscore,trainscore,testscore,test_pred,con_matrix,class_report)