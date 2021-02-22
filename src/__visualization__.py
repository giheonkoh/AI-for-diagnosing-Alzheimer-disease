def SMTM(history, results) : 
    import matplotlib.pyplot as plt

    acctrain = [s * 100 for s in history.history['accuracy']]
    plt.plot(acctrain,'r-')

    accval = [s * 100 for s in history.history['val_accuracy']]
    plt.plot(accval,'b-')

    losstrain = [l * 20 for l in history.history['loss']]
    plt.plot(losstrain,'r--')

    lossval = [l *20 for l in history.history['val_loss']]
    plt.plot(lossval, 'b--' )

    plt.title('scaled model accuracy {}/ loss {}'.format(round(results[1],2),round(results[0],2)))
    plt.ylabel('scale (%)')
    plt.xlabel('epochs')
    plt.legend(['train_accuracy','test_accuracy','train_loss (scaled)', 'test_loss (scaled)'],loc = 'center right')

    plt.show()
