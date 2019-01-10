from sklearn import preprocessing
from sklearn import linear_model
from sklearn import neural_network

def classify_with_logistic_regression(X, y):
  # Perform classification using continous variables
  standard_scaler = preprocessing.StandardScaler()
  X = standard_scaler.fit_transform(X)

  logistic_regressor = linear_model.LogisticRegression(solver='lbfgs')
  print("Training Logistic Regressor...")
  logistic_regressor.fit(X, y)

  print("Logistic regression score (USING TRAINING SET): ",logistic_regressor.score(X, y))

def classify_with_nn(X, y):
  hidden_layer_sizes = (10, 10, 8, 6, 4)

  standard_scaler = preprocessing.StandardScaler()
  X = standard_scaler.fit_transform(X)

  clf = neural_network.MLPClassifier(activation='relu',
                                     batch_size='auto',
                                     hidden_layer_sizes=hidden_layer_sizes,
                                     learning_rate='adaptive',
                                     learning_rate_init=0.0001,
                                     max_iter=2000,
                                     momentum=0.9,
                                     n_iter_no_change=100,
                                     random_state=1,
                                     shuffle=True,
                                     #solver='lbfgs',
                                     solver='adam',
                                     verbose=False,
                                     warm_start=False)
  print("Training NN...")
  clf.fit(X, y)
  print("NN score (USING TRAINING SET): ", clf.score(X, y))
  
