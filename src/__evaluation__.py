def kfold(k = 5) :
    from sklearn.model_selection import KFold
    return KFold(n_splits = k, shuffle = True, random_state = 42)

def report_KfoldCV(model, X, y, k, kfold, callbacks) :
    from sklearn.model_selection import cross_val_score
    return cross_val_score(model, X, y, cv = kfold, fit_params = {'callbacks': callbacks})

def minimize_Epoch(patience = 0) :
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    return [EarlyStopping(monitor = 'val_accuracy', mode = 'max', patience = patience, verbose =0)]

def getROCAUC(ytest, ypred) :
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(ytest, ypred)
    return auc

def getConfMatrix(ytest, ypred) :
    from sklearn.metrics import confusion_matrix
    conf = confusion_matrix(ypred.round(), ytest)
    return conf

def getAcc(ytest, ypred) :
    from sklearn import metrics
    return metrics.accuracy_score(ytest, ypred)

def getReport(y, ypred) :
    from sklearn.metrics import classification_report
    print(classification_report(y,ypred))

def translate(diag, what = 'adcn') :
    if what == 'adcn' :
        if diag == 1 :
            return 'Dementia'
        elif diag == 0 :
            return 'Normal'
