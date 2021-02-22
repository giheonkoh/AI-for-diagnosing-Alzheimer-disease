#V1: Build binary classification MLP-DL MainFrame
#V2: For every epoch, present the variation of loss in terms of standard deviation and mean
#V3: Added more layer and activation function/ develop model effectiveness: test_acc = 0.90, auc = 0.85 --> Overfitting
#V4: tested with more layers, epochs, and activation functions: Overfitting -> good fit, validation_acc = 0.90, test_acc = 0.90, auc = 0.899 at epoch = 35
#V4+:   added manual backpropagation method to check which factor would affect more.
#V5: added the automated model vs validation graph for accuracy and loss comparison.
#V6: test with the new dataset
#V6+:   remove 52 features according to Ref (=TBD). good fit, validation_acc = 0.95, test_acc = 0.96, auc = 0.94 at epoch = 40
#V6+:   added t-test to remove more columns --> Declined --> to be tested based on FDR
#V7: Tensorflow version --> runs Faster and shows higher accuracy for a reason.
#V8: Newly added data with Kfold algorithm
#V9: Code Capsulization --> For further maintainance
#V10: save the model as HDF5 file
#V11: Add RF and GBT classes

"""
01_N : Siemense Skyra 3T
01_P : Siemense Skyra 3T
02_P : Philips Ingenia 3.0T --> Too many NaN values
03_P : Philips Achieva 3.0T
04_P : GE DISCOVERY MR750w --> No Normal Data
05_N : Philips Achieva 3.0T
06_N : Philips Achieva 3.0T
"""

def TRAIN(file, mri) :

    print("*** UPDATE MODULE *** ")

    #import packages
    import __preprocessing__ as prep
    import __model__ as model
    import __evaluation__ as eval
    import __visualization__ as viz

    #call in
    #data = prep.getdata(file)
    data = prep.filtercolumns(prep.getdata(file).dropna(axis = 1))


    """#Feature Selection (최초 1회 실행 후 사용 X)
    data = prep.FeatureSelection(data, prob = 0.1) #0.1(acc, auc) > 0.05(acc, auc)
    cols = data.columns.tolist()
    cols.remove('y')
    prep.saveColumns(cols, "selected_cols.csv")"""

    cols = prep.getColumns('data/selected_cols.csv')
    data = data[cols.tolist() + ['label']]

    #preprocessing
    X,y = prep.xysplit(data)
    X = prep.normalization(X, how = 'minmax')

    #Data Split
    xtrain,xtest,ytrain,ytest = prep.split_test_train(X[:-3], y[:-3], 0.3)

    #modeling
    ########################### MLP ###########################
    fn = model.MLP()
    fn.model_design(batch_size = X.shape[1], prob_to_drop = 0.2)
    optim = model.get_optim(lr = 0.001)
    fn.compile(optimizer = optim)
    classifier = fn.Keras_fy(fn = fn.build_fn(), epochs = 100, val_split = 0.2)
    fn.summarize_model()

    #Plug In
    callbacks = eval.minimize_Epoch(patience = 15)

    history = fn.showhistory(xtrain, ytrain, callbacks = callbacks)

    #evaluation
    print("\n\n** DL - MLP Classifier **")
    results = fn.test(xtest, ytest)
    print("\nloss: {} \nacc: {}".format(results[0],results[1]))

    ypred = fn.predictMLP(xtest)

    print("\nAUC: {}".format(eval.getROCAUC(ytest, ypred)))
    print("\nconfusion matrix: \n{}".format(eval.getConfMatrix(ytest, ypred)))

    #plotting
    #viz.SMTM(history, results)

    ############################ RF ###########################
    print("\nUpdating RF Model....")

    #parameter definition
    parameter = {
        'n_estimators' : [20, 50, 100],
        'max_depth' : [10, 20],
        'min_samples_leaf' : [3,5,7],
        'min_samples_split' : [3,5,7]
    }

    rf = model.Tree('rf', n_estimator = 10)
    rf.apply_GridSearchCV(param = parameter)
    rf.trainTree(xtrain, ytrain)
    rf.show_best_one() #100/ 20/ 5/ 3
    ypred = rf.predictTree(xtest)
    print("\n\n** Ramdom Forest Classifier **")
    print("\nacc: {} \nAUC: {}".format(eval.getAcc(ytest, ypred), eval.getROCAUC(ytest, ypred)))
    print("confusion matrix: \n{}\n\n".format(eval.getConfMatrix(ytest, ypred)))

    ########################### GBT ###########################
    print("Updating GBT Model....")

    #parameter definition
    parameter = {
        'n_estimators' : [50, 70, 100],
        'max_depth' : [5, 10],
        'min_samples_leaf' : [5,7],
        'min_samples_split' : [3, 5],
        'learning_rate' : [0.01,0.001]
    }

    gbt = model.Tree('gbt',n_estimator = 10)
    gbt.apply_GridSearchCV(param = parameter)
    gbt.trainTree(xtrain, ytrain)
    gbt.show_best_one() #100/ 5/ 7/ 3/ 0.01
    ypred = gbt.predictTree(xtest)
    print("\n\n** Gradient Boosting Tree Classifier **")
    print("\nacc: {} \nAUC: {}".format(eval.getAcc(ytest, ypred), eval.getROCAUC(ytest, ypred)))
    print("confusion matrix: \n{}".format(eval.getConfMatrix(ytest, ypred)))

    #############################################################

    fn.save_model('savefile/MLP.h5')
    model.saveTree(rf, "savefile/RF.sav")
    model.saveTree(gbt, "savefile/GBT.sav")

def TEST(file, mri) :
    print("*** TEST MODULE ***")

    import __preprocessing__ as prep
    import __model__ as model
    import __evaluation__ as eval

    #Match Columns
    cols = prep.getColumns('data/selected_cols.csv')

    #Import Data
    data = prep.getdata(file)[cols]
    ref = prep.getdata('data/{}_minmax.csv'.format(mri))[cols].iloc[-2:,:] #Min/ Max

    data = prep.con(data, ref) #To concat

    X =  prep.normalization(data, how = 'minmax')[0].reshape(1,-1)

    #Load Model
    #mlp
    mlp = model.load_model('savefile/MLP.h5')
    ypred_mlp = mlp.predictMLP(X)

    #rf
    rf = model.loadTree('savefile/RF.sav')
    ypred_rf = rf.predictTree(X)

    #gbt
    gbt = model.loadTree('savefile/GBT.sav')
    ypred_gbt = gbt.predictTree(X)

    #print and compare answers
    print("RESULTS: \n Answers from MLP: {}\n\n Answers from RF: {}\n\n Answers from GBT: {}".format(eval.translate(ypred_mlp.flatten()), eval.translate(ypred_rf), eval.translate(ypred_gbt)))

    #print report
    res = [eval.translate(ypred_mlp.flatten()), eval.translate(ypred_rf), eval.translate(ypred_gbt)]
    if len(list(set(res))) == 1 :
        print('\n\n*** You are {} !!! *** \n\n'.format(max(set(res), key = res.count)) )
    else :
        print('\n\n*** You are diagnosed with {} but might be MCI. Go to your doctor for further details !!! *** \n\n'.format(max(set(res), key = res.count)) )


#TRAIN AND TEST MODULES
def op(filename, mri, what) :
    MRI_machines = ['sm', 'pl', 'ge'] #simmens, Philips, GE

    if mri in MRI_machines :
        if what == 'train' :
            TRAIN(filename, mri)

        elif what == 'test' :
            TEST(filename, mri)

        else :
            print('!!! INSERT EITHER \"{}\" OR \"{}\" !!!'.format('train', 'test'))
    else :
        print('!!! PLEASE INPUT THE PROPER MRI DEVICE CODE !!!')
