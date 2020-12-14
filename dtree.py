import pandas as pd
import pydotplus as py
import matplotlib.pyplot as pl
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import pickle


def read_csv():
    col_names = ['Day', 'Month', 'Year', 'mean_temp', 'max_temp',
                 'min_temp', 'meanhum', 'meandew', 'pressure', 'heat', 'wet']
    pd.set_option('display.max_rows', None)
    data = pd.read_csv(
        'C:\\Users\\tmquang\\Downloads\\data.csv', index_col=None)
    return data


def set_values(inputs):

    data = read_csv()
    # print(data.head())
    heat = {'YES': 1, 'NO': 0}
    wet = {'YES': 1, 'NO': 0}
    data['heat'] = data['heat'].map(heat)
    data['wet'] = data['wet'].map(wet)
    #
    # ind_vals = ['Day', 'Month', 'Year'
    #     , 'mean_temp', 'max_temp', 'min_temp'
    #     , 'meanhum', 'meandew', 'pressure']
    # x = data[ind_vals]label
    # d_vals = ['wet']
    # y = data[d_vals]
    feature_cols = ['Day', 'Month', 'Year', 'mean_temp',
                    'max_temp', 'min_temp', 'meanhum', 'meandew', 'pressure']
    X = data[feature_cols]  # Features
    y = data.wet  # Target variable

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifier = DecisionTreeClassifier(splitter='random', max_depth=3)
    clf = classifier.fit(x_train, y_train)
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(classifier, open(filename, 'wb'))
    test = pd.DataFrame.from_dict(inputs)
    #test = pd.DataFrame(x_test.iloc[:1, :])
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    # print(test)
    # print(type(x_test))
    # print(x_test.iloc[0])
    y_pred = clf.predict(test)
    # print(y_pred)
    # print(y_test)

    accuracy = loaded_model.score(x_test, y_test)
    # print(y_pred)
    print(accuracy)
    # print(confusion_matrix(y_test, y_pred))
    #
    # print(classification_report(y_test, y_pred))
    # print('')
    text_representation = tree.export_text(classifier)
    # print(text_representation)
    # print('')
    # print(classifier.classes_)
    return [accuracy, y_pred]


#data = set_values()


# def fig():
#     fig = pl.figure(figsize=(25, 20))
#     _ = tree.plot_tree(classifier)
#     fig.savefig("decistion_tree.png")


# fig()
