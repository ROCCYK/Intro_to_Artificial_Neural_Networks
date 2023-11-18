import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

mlflow.set_experiment('classification_model')

iris = datasets.load_iris()
x = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)

with mlflow.start_run(run_name='iris-model-experiment') as run:
    
	# add parameters for tuning
	num_estimators = 100
	mlflow.log_param('num_estimators', num_estimators)

	# train the model
	rf = RandomForestClassifier(n_estimators=num_estimators)
	rf.fit(X_train, y_train)
	predictions = rf.predict(X_test)

	mlflow.sklearn.log_model(rf, 'random-forest-model')

	# log model performance 
	accuracy = accuracy_score(y_test, predictions)
	mlflow.log_metric('Accuracy', accuracy)
	print('Accuracy: %f' % accuracy)

	run_id = run.info.run_uuid
	experiment_id = run.info.experiment_id
	mlflow.end_run()
	print(f'artifact_uri = {mlflow.get_artifact_uri()}')
	print(f'runID: {run_id}')