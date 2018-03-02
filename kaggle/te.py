import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.callbacks import ModelCheckpoint
from tensorflow import set_random_seed
np.random.seed(42)
set_random_seed(42)


def PreprocessData(raw_df):
    df = raw_df.drop(['Train_ID'], axis = 1)

    # df = df.drop(['SEX'], axis = 1)

    # df['EDUCATION'] = df['EDUCATION'].map({0:0, 1:1, 2:2, 3:3, 4:0, 5:0, 6:0})

    df['PAY_1'] = df['PAY_1'].map({-2:-2, -1:-1, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:6, 8:6, 9:6})


    # x_OneHot_df = pd.get_dummies(data = df, columns = ['MARRIAGE'])
    # x_OneHot_df = x_OneHot_df.drop(['MARRIAGE_0'], axis = 1)

    # x_OneHot_df = pd.get_dummies(data = df, columns = ['EDUCATION'])
    # x_OneHot_df = x_OneHot_df.drop(['EDUCATION_0'], axis = 1)

    x_OneHot_df = pd.get_dummies(data = df, columns = ['PAY_1'])
    x_OneHot_df = x_OneHot_df.drop(['PAY_1_6'], axis = 1)


    feature = x_OneHot_df.drop(['Y'], axis = 1)
    label = x_OneHot_df['Y']


    Label = label.values
    Features = feature.values

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

    scaledFeatures = minmax_scale.fit_transform(Features)

    return scaledFeatures, Label




if __name__ == '__main__':
    all_df = pd.read_csv(sys.argv[1])

# all_df = pd.read_csv('/Users/jason18150/Documents/AI_HW/kaggle/Train.csv')




    train_Features, train_Label = PreprocessData(all_df)

    model = Sequential()

    model.add(Dense(units = 200, input_dim = 30, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 175, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(0.3, seed = 42))
    model.add(Dense(units = 75, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(0.3, seed = 42))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    train_history = model.fit(x = train_Features, y = train_Label, validation_split = 0.1, epochs = 30, batch_size = 30, callbacks=callbacks_list, verbose=0)

    # scores = model.evaluate(x = test_Features, y = test_Label)

    scores = model.evaluate(train_Features, train_Label, verbose=0)
    print scores[1]*100

    # all_Features, Label = PreprocessData(all_df)
    model.save('model.h5')



