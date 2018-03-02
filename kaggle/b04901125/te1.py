import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model



def PreprocessTestData(raw_df):
    df = raw_df.drop(['Test_ID'], axis = 1)

    # df = df.drop(['SEX'], axis = 1)

    # df['EDUCATION'] = df['EDUCATION'].map({0:0, 1:1, 2:2, 3:3, 4:0, 5:0, 6:0})

    df['PAY_1'] = df['PAY_1'].map({-2:-2, -1:-1, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:6, 8:6, 9:6})

    # x_OneHot_df = pd.get_dummies(data = df, columns = ['MARRIAGE'])

    # x_OneHot_df = x_OneHot_df.drop(['MARRIAGE_0'], axis = 1)

    # x_OneHot_df = pd.get_dummies(data = df, columns = ['EDUCATION'])
    # x_OneHot_df = x_OneHot_df.drop(['EDUCATION_0'], axis = 1)

    x_OneHot_df = pd.get_dummies(data = df, columns = ['PAY_1'])

    x_OneHot_df = x_OneHot_df.drop(['PAY_1_0'], axis = 1)


    ndarray = x_OneHot_df.values


    # print ndarray[:2]

    Features = ndarray

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

    scaledFeatures = minmax_scale.fit_transform(Features)

    return scaledFeatures



if __name__ == '__main__':
    model = load_model('model.h5')

    test_public_df = pd.read_csv(sys.argv[1])
    test_private_df = pd.read_csv(sys.argv[2])


    test_Features = PreprocessTestData(test_public_df)
    test_probability = model.predict(test_Features)

    p = test_public_df
    p.insert(len(test_public_df.columns), 'probability', test_probability)

    lc = pd.DataFrame(data = p)

    lc = lc.rename(columns = {'Test_ID': 'Rank_ID'})


    d = lc.sort_values(by = ['probability'], ascending = False)

    Rank = d.iloc[:, 0]

    Rank.to_csv('public.csv', index = None, header = 'Rank_ID')



    test_Features = PreprocessTestData(test_private_df)
    test_probability = model.predict(test_Features)

    p = test_private_df
    p.insert(len(test_private_df.columns), 'probability', test_probability)

    lc = pd.DataFrame(data = p)

    lc = lc.rename(columns = {'Test_ID': 'Rank_ID'})


    d = lc.sort_values(by = ['probability'], ascending = False)

    Rank = d.iloc[:, 0]

    Rank.to_csv('private.csv', index = None, header = 'Rank_ID')


