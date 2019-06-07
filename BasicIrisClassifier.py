from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from joblib import dump, load

class BasicIrisClassifier:
  def load(self):
    iris = load_iris()
    self.data = iris.data
    self.target = iris.target
    self.target_names = iris.target_names

  def train(self):
    data_train, data_test, target_train, target_test = train_test_split(self.data, self.target, test_size=0.3, random_state=12)

    self.classifier = KNeighborsClassifier()
    self.classifier.fit(data_train, target_train)

    target_pred = self.classifier.predict(data_test)
    accuracy = metrics.accuracy_score(target_test, target_pred)

    return accuracy

  def predict(self, external_input_sample):
    prediction_raw_values = self.classifier.predict(external_input_sample)
    prediction_resolved_values = [self.target_names[p] for p in prediction_raw_values]
    return prediction_resolved_values

  def saveModel(self):
    dump(self.classifier, 'trained_iris_model.pkl')
    dump(self.target_names, 'trained_iris_model_targetNames.pkl')

  def loadModel(self):
    self.classifier = load('trained_iris_model.pkl')
    self.target_names = load('trained_iris_model_targetNames.pkl')

# Using BasicIrisClassifier
external_input_sample = [[5, 2, 4, 1], [6, 3, 5, 2], [5, 4, 1, 0.5]]
basic_iris_classifier = BasicIrisClassifier()

basic_iris_classifier.load()

accuracy = basic_iris_classifier.train()
print("Model Accuracy:", accuracy)

prediction = basic_iris_classifier.predict(external_input_sample)
print("Prediction for {0} => \n{1}".format(external_input_sample, prediction))

basic_iris_classifier.saveModel()
#basic_iris_classifier.loadModel()
