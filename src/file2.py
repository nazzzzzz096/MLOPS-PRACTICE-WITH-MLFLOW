import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='nazzzzzz096', repo_name='MLOPS-PRACTICE-WITH-MLFLOW', mlflow=True)


mlflow.set_tracking_uri("https://dagshub.com/nazzzzzz096/MLOPS-PRACTICE-WITH-MLFLOW.mlflow")


# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 8
n_estimators = 11
# mention you experiment 

mlflow.set_experiment('experiment-1')

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(X_train,y_train)


    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)
    
    #creating confusin matrix

    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('predicted')
    plt.title('confusion_matrix')

    plt.savefig("confusion-matrix.png")

    #log artifacts using mlflow

    mlflow.log_artifact("confusion-matrix.png")
    mlflow.log_artifact(__file__)


    #setting tag 
    mlflow.set_tags({"author":"nazina","project":"wine data"})


    #log the model
    mlflow.sklearn.log_model(rf,artifact_path="rf_model", registered_model_name=None)
    




    print(accuracy)
