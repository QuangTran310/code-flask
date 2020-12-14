from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from dtree import set_values
#from script import database

app = Flask(__name__)


'''@app.route("/")
def index():
    to_send = database()
    return render_template("index.html", to_send=to_send)'''


'''@app.route('/project', methods=['GET', 'POST'])
def project():
    if request.method == 'POST':
        datepicker = request.form['datepicker']
        datetime_object = datetime.strptime(datepicker, '%Y-%m-%d')
        print(datetime_object)
        print(datetime_object.month)
        print(datetime_object.day)
        print(datetime_object.year)
        print(datepicker)
        meantemp = request.form['meantemp']
        print(meantemp)
        maxtemp = request.form['maxtemp']
        print(maxtemp)
        mintemp = request.form['mintemp']
        print(mintemp)
        minhumidity = request.form['minhumidity']
        print(minhumidity)
        meandew = request.form['meandew']
        print(meandew)
        pressure = request.form['pressure']
        print(pressure)'''


@app.route('/project', methods=['GET', 'POST'])
def input_vals():
    if request.method == 'POST':
        datepicker = request.form['datepicker']
        datetime_object = datetime.strptime(datepicker, '%Y-%m-%d')
        day = datetime_object.day
        month = datetime_object.month
        year = datetime_object.year
        meantemp = request.form['meantemp']
        maxtemp = request.form['maxtemp']
        mintemp = request.form['mintemp']
        minhumidity = request.form['minhumidity']
        meandew = request.form['meandew']
        pressure = request.form['pressure']
        print(day, month, year, meantemp, maxtemp,
              mintemp, minhumidity, meandew, pressure)
        '''# inputs = [day, month, year, meantemp, maxtemp,
        #           mintemp, minhumidity, meandew, pressure]
        # new_data = []
        # test = pd.DataFrame(x_test.iloc[:1, :])
        objects = {
            "day": day,
            "month": month,
            "year": year,
            "meantemp": meantemp,
            "maxtemp": maxtemp,
            "mintemp": mintemp,
            "minhumidity": minhumidity,
            "meandew": meandew,
            "pressure": pressure,
        }

        # objects = {
        #     "day": 1,
        #     "month": 5,
        #     "year": 2020,
        #     "meantemp": 111,
        #     "maxtemp": 11,
        #     "mintemp": 11,
        #     "minhumidity": 11,
        #     "meandew": 11,
        #     "pressure": 11
        # }

        data = set_values(objects)
        # for i in input:
        #     new_data.append(i)
        # user_data = pd.DataFrame(new_data)
        # print(user_data)'''
        return render_template('Project.html')
    else:
        return render_template('Project.html')


# @app.route("/knn", methods = ["GET", "POST"])
'''def knn_implement():
    if request.method == 'POST':

        data = pd.read_csv('C:\\Users\\tmquang\\Downloads\\data.csv')
        heat = {'YES': 1, 'NO': 0}
        wet = {'YES': 1, 'NO': 0}
        data['heat'] = data['heat'].map(heat)
        data['wet'] = data['wet'].map(wet)

        ind_vals = ['Day', 'Month', 'Year', 'mean_temp',
            'max_temp', 'min_temp', 'meanhum', 'meandew', 'pressure']
        x = data[ind_vals]
        d_vals = ['wet']
        y = data[d_vals]

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, shuffle=True)

        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(X_train, y_train.values.ravel())
        y_pred = classifier.predict(X_test)
        acc = classifier.score(X_test, y_test)
        print(acc)

        print(confusion_matrix(y_test, y_pred))

        print(classification_report(y_test, y_pred))

    return render_template('Project.html')'''
