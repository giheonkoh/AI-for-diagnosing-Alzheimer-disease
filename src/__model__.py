# MLP
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

class MLP() :
    def __init__(self) :
        self.model = Sequential()
        self.batch_size = 0
        self.prob_to_drop = 0

    def model_design(self, batch_size, prob_to_drop) :
        self.batch_size = batch_size
        self.prob_to_drop = prob_to_drop
        self.model.add(Dense(units = self.batch_size, activation = 'swish', input_dim = self.batch_size))
        self.model.add(Dropout(self.prob_to_drop))
        self.model.add(Dense(units = self.batch_size, activation = 'swish'))
        self.model.add(Dropout(self.prob_to_drop))
        self.model.add(Dense(units = self.batch_size, activation = 'swish'))
        self.model.add(Dropout(self.prob_to_drop))
        self.model.add(Dense(units = self.batch_size, activation = 'swish'))
        self.model.add(Dropout(self.prob_to_drop))
        self.model.add(Dense(units = 1, activation = 'sigmoid'))

    def build_fn(self) : #For function definition only! No further usage!
        m = Sequential()
        m.add(Dense(units = self.batch_size, activation = 'swish', input_dim = self.batch_size))
        m.add(Dropout(self.prob_to_drop))
        m.add(Dense(units = self.batch_size, activation = 'swish'))
        m.add(Dropout(self.prob_to_drop))
        m.add(Dense(units = self.batch_size, activation = 'swish'))
        m.add(Dropout(self.prob_to_drop))
        m.add(Dense(units = self.batch_size, activation = 'swish'))
        m.add(Dropout(self.prob_to_drop))
        m.add(Dense(units = 1, activation = 'sigmoid'))

        #model compiler
        m.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        #Returning model
        return m

    def compile(self, optimizer) :
        self.model.compile(optimizer= optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    def showhistory(self, xtr, ytr, epochs = 100, val_split = 0.2, callbacks = None) :
        return self.model.fit(xtr, ytr, epochs = epochs, batch_size=self.batch_size, validation_split = val_split, callbacks = callbacks)

    def test(self, xtest, ytest) :
        results = self.model.evaluate(xtest, ytest, verbose = 0)
        return results

    def Keras_fy(self, fn = None, epochs = 100, val_split = 0.2) :
        return KerasClassifier(build_fn = fn, epochs = epochs, batch_size = self.batch_size, verbose = 0, validation_split = val_split)

    def summarize_model(self) :
        return self.model.summary()

    def predictMLP(self, xtest) :
        #ypred = self.model.predict_classes(xtest).flatten()
        ypred = (self.model.predict(xtest) > 0.5).astype("int32")
        return ypred

    def save_model(self, name) :
        self.model.save(name)
        print("Model Saved!")

    def checkpoint(self) :
        return print('\n\n!! Pinned Up Here !!\n\n')

def get_optim(lr) :
    return tf.keras.optimizers.Adam(learning_rate = lr)

def get_loss(pred, y) :
    return tf.losses.binary_crossentropy(pred, y)

def load_model(loc) :
    new_model = MLP()
    new_model.model = tf.keras.models.load_model(loc)
    #print(new_model.summarize_model())
    return new_model


# RF & GBT
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle5 as pickle

class Tree() :
    def __init__(self, what, n_estimator) :
        if what == 'gbt' :
            self.clf = GradientBoostingClassifier(n_estimators = n_estimator, random_state = 42)
        elif what == 'rf' :
            self.clf = RandomForestClassifier(n_estimators = n_estimator, random_state = 42)
        else :
            print("!!! Insert either 'gbt' OR 'rf' !!! ")

    def trainTree(self, xtrain, ytrain) :
        self.clf.fit(xtrain, ytrain)

    def apply_GridSearchCV(self, param) :
        from sklearn.model_selection import GridSearchCV
        self.clf = GridSearchCV(self.clf, param_grid=param, scoring="accuracy", cv=3, verbose=1, n_jobs=-1)

    def show_best_one(self) :
        print('\nBest Hyperparameter: ', self.clf.best_params_)
        print('Best Hyperparameter\'s accuracy: {0:.3f}'.format(self.clf.best_score_))
        self.clf = self.clf.best_estimator_

    def predictTree(self, xtest) :
        return self.clf.predict(xtest)

def saveTree(tree, filename) :
    pickle.dump(tree, open(filename, 'wb'))
    print("Model Saved!")

def loadTree(filename) :
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
