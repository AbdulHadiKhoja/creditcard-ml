import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("creditcard.csv")

def printScores(train,test,pred):
    print(f"TrainScore: {train}")
    print(f"TestScore: {test}")
    print(f"Predict: {pred}")

X = df.drop("Class",axis=1)
y = df["Class"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)
model = RandomForestClassifier(random_state=42,n_estimators=100)
model.fit(X_train,y_train)

pred = model.predict(X_test.iloc[[0]])
trainscore = model.score(X_train,y_train)
testscore = model.score(X_test,y_test)

printScores(trainscore,testscore,pred)