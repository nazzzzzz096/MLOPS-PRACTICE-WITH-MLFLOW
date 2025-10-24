import mlflow
print("printing tracking URI schema below")
print(mlflow.get_tracking_uri())
print("\n")


mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("printing tracking URI schema below")
print(mlflow.get_tracking_uri())
print("\n")