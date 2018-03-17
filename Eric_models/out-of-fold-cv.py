from sklearn.metrics import roc_auc_score
K_fold=split
pred_val_accumulator=test_accumulator=None
accumulator=[]

#change according to the John's version, but according to the wanring message, I do anthor version
#My modified version
from sklearn.model_selection import KFold

kf = KFold(n_splits=10,shuffle=True,random_state=42)

for train_index, val_index in kf.split(np.zeros(train.shape[0])):
    c_train_x=X_tr[train_index]
    c_train_y=y_tr[train_index]
    c_val_X = X_tr[val_index]
    c_val_y = y_tr[val_index]
     
    
    
#John's version (still in the competition for training)
class_names = list(train)[-6:]
multarray = np.array([100000, 10000, 1000, 100, 10, 1])
y_multi = np.sum(train[class_names].values * multarray, axis=1)

print(class_names)

print(y_multi)


from sklearn.model_selection import StratifiedKFold
splits = 10
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

train_ids = [] 
val_ids = []
for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(train.shape[0]), y_multi)):
    train_ids.append(train.loc[train_idx, 'id'])
    val_ids.append(train.loc[val_idx, 'id'])

for i in range(splits):
    print("=================================")
    print("Start on: "+str(i)+" fold")
    c_train_X = X_tr[train.id.isin(train_ids[i])]
    c_train_y = y_tr[train.id.isin(train_ids[i])]
    c_val_X = X_tr[train.id.isin(val_ids[i])]
    c_val_y = y_tr[train.id.isin(val_ids[i])]
    
    
    
    
    
    
    
        
        
    print("")
    #this part can also change to other model, logreg, svc, lgbm,....etc.
    file_path="weights_base_5_fold_CNN.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min",verbose=1, patience=5)
    callbacks_list = [early,checkpoint]
    model = get_model()
    seed(1)   
    history=model.fit(c_train_X,c_train_y, batch_size=batch_size, epochs=epochs, 
                      validation_data=(c_val_X, c_val_y), callbacks=callbacks_list) #[c_train_X, c_train_X_gram]
    #history=model.fit({'main_input':c_train_X, 'aux_input': c_train_X_twitter},c_train_y, batch_size=batch_size, epochs=epochs, 
                      #validation_data=({'main_input':c_val_X, 'aux_input': c_val_X_twitter}, c_val_y),callbacks=callbacks_list)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


  
     ##code validation for NN model
    print(history.history.keys())
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.savefig('k-fold-plot1'+str(i)+'.png', format='png')

 
    print(history.history.keys())
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.savefig('k-fold-plot2'+str(i)+'.png', format='png')
    
    model.load_weights(file_path)
    pred_val = model.predict(c_val_X,batch_size=batch_size, verbose=1)
    #pred = model.predict( {'main_input':c_val_X, 'aux_input': c_val_X_twitter},batch_size=batch_size, verbose=1)
    y_test = model.predict( X_te_1,batch_size=batch_size, verbose=1)
   
    if(i==1):
         pred_val_accumulator=pred_val
         test_accumulator=y_test
    else:
         pred_val_accumulator=np.vstack((pred_val_accumulator,pred_val))
         test_accumulator=test_accumulator+y_test
    sub_accumulator=[]
    for j in range(0,len(list_classes)):
        result=pred_val[:,j].reshape(-1, 1)
        roc_score=roc_auc_score(c_val_y[:,j].reshape(-1, 1),result)
        print("#Column: "+str(j)+" Roc_auc_score: "+str(roc_score))
        sub_accumulator.append(roc_score)
    print("#Average Roc_auc_score is: {}\n".format( np.mean(sub_accumulator) ))
    #prevent the program break down(core dump), save to the disk at each step
    pickle.dump(pred_val_accumulator,open("prediction_RNN_1_"+str(i)+".pkl", "wb"))
    pickle.dump(test_accumulator,open("test_average_1_"+str(i)+".pkl", "wb"))
    accumulator.append(np.mean(sub_accumulator))
    del model
test_average=test_accumulator/K_fold

print("#Total average Roc_auc_score is: {}\n".format( np.mean(accumulator) ))    


pickle.dump(pred_val_accumulator,open("prediction_RNN.pkl", "wb"))
if test_accumulator is not None:
   pickle.dump(test_average,open("test_average.pkl", "wb"))
