from flask import Flask, render_template, request, session, url_for
import pandas as pd
import numpy as np
from werkzeug.utils import redirect
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import csv
from datetime import datetime
from datetime import date
from dateutil.parser import parse
import string
from nltk.corpus import stopwords
import nltk
from collections import OrderedDict
import re
import pygal

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload',methods=["POST","GET"])
def uploaddataset_csv_submitted():
    if request.method == "POST":
        csvfile = request.files['csvfile']
        result = csvfile.filename
        file = "D:/Need to execute final projects/A Framework for Real-Time Spam Detection in Twitter/CODE/" + result
        print(file)
        session['filepath'] = file


        return render_template('uploaddataset.html',msg='sucess')
    return render_template('uploaddataset.html')


@app.route('/viewdata',methods=["POST","GET"])
def viewdata():
    session_var_value = session.get('filepath')
    df = pd.read_csv(session_var_value,encoding='latin-1')
    #print(df)
    x = pd.DataFrame(df)

    return render_template("view.html", data=x.to_html())

@app.route('/preprocess',methods=["POST","GET"])
def preprocessdata():
    session_var_value = session.get('filepath')
    df = pd.read_csv(session_var_value,encoding='latin-1')
    #print(df)
    x = pd.DataFrame(df)
    data = df.drop(['latitude', 'longitude', 'source_r', 'emoji_names', 'location'], axis=1)
    data.to_csv("D:\\Need to execute final projects\\A Framework for Real-Time Spam Detection in Twitter\\CODE\\preprocess.csv")
    users = {}
    with open('D:\\Need to execute final projects\\A Framework for Real-Time Spam Detection in Twitter\\CODE\\twitter.csv', 'r') as infile:

        rows = csv.reader(infile)
        for row in rows:
            if (not (row[1] == 'created')):
                date1 = datetime.date(parse(row[1].replace('/', '-')))
                date2 = datetime.date(datetime.now())
                delta = date2 - date1
                days = delta.days
                users[row[0]] = days
    print(users)
    with open('D:\\Need to execute final projects\\A Framework for Real-Time Spam Detection in Twitter\\CODE\\consolidated.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(
            ['Tweet Id', 'Account Age', 'No. Follower', 'No. Following', 'No. Userfavourites', 'No. Tweets',
             'No. Retweets', 'No. Hashtags', 'No. Url', 'No. Char', 'No. Digits', 'No. Non Ascii'])
        tweetid = df.groupby('tweetid')['followers'].sum()
        followers = df.groupby('tweetid')['followers'].sum()
        following = df.groupby('tweetid')['following'].sum()
        favourites = df.groupby('tweetid')['favorites'].sum()
        hashtag = df.groupby('tweetid')['hashtag'].count()
        tweets = df.groupby('tweetid')['tweetid'].count()
        retweets = df.groupby('tweetid')['retweets'].sum()
        urls = df.groupby('tweetid')['url'].count()
        textchars = df.groupby('tweetid')['text'].nth(0)
        length = []
        digits = []
        alpha = []
        asci = []
        for text in textchars:
            dcount = 0
            acount = 0
            ascount = 0
            for char in text:
                if char.isdigit():
                    dcount += 1
                elif char.isalpha():
                    acount += 1
                elif char.isspace():
                    pass
                else:
                    ascount += 1
            digits.append(dcount)
            alpha.append(acount)
            asci.append(ascount)
        # print(digits)
        # print(alpha)
        # print(asci)

        norecords = len(tweetid)
        print(norecords)
        index = 0
        keys = tweetid.keys()
        for i in keys:
            row = []
            row.append(i)
            row.append(users.get(str(i)))
            row.append(followers.get(i))
            row.append(following.get(i))
            row.append(favourites.get(i))
            row.append(tweets.get(i))
            row.append(retweets.get(i))
            row.append(hashtag.get(i))
            row.append(urls.get(i))
            row.append(alpha[index])
            row.append(digits[index])
            row.append(asci[index])
            index += 1
            filewriter.writerow(row)
    x = pd.read_csv("D:\\Need to execute final projects\\A Framework for Real-Time Spam Detection in Twitter\\CODE\\consolidated.csv")
    return render_template("preprocess.html", data=x.to_html())

@app.route('/topspamham',methods=["POST","GET"])
def topspamham():
    cons = pd.read_csv("D:\\Need to execute final projects\\A Framework for Real-Time Spam Detection in Twitter\\CODE\\twitter.csv", encoding='latin-1')
    spam = cons['text' and cons['class'] == 'spam']
    spam = spam['text']
    ham = cons['text' and cons['class'] == 'ham']
    ham = ham['text']
    hamtext = ''
    for text in ham:
        hamtext += text
    spamtext = ''
    for text in spam:
        spamtext += text
    spamtext = spamtext.replace("[^\w\s]", "").lower()
    hamtext = hamtext.replace("[^\w\s]", "").lower()
    spamtext = ' '.join([word for word in spamtext.split() if word not in (stopwords.words('english'))])
    spamtext = re.sub(r'[^\w\s]', '', spamtext)
    hamtext = ' '.join([word for word in hamtext.split() if word not in (stopwords.words('english'))])
    hamtext = re.sub(r'[^\w\s]', '', hamtext)
    word2count = {}
    hamwords = hamtext.split(" ")
    for word in hamwords:
        if (len(word) >= 4):
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1
    # print(word2count)
    od = OrderedDict(sorted(word2count.items(), key=lambda x: x[1]))
    count = 0
    length = len(od)
    tophamwords = ['TOP HAM WORDS']
    for item in od.items():
        count += 1
        if (count > length - 30):
            tophamwords.append(item[0])
    tophamwords.append(' ')
    tophamwords.append(' ')

    tophamdf = pd.DataFrame(tophamwords)
    word2count = {}
    spamwords = spamtext.split(" ")
    for word in spamwords:
        if (len(word) >= 4):
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1
    # print(word2count)
    od = OrderedDict(sorted(word2count.items(), key=lambda x: x[1]))
    count = 0
    length = len(od)
    topspamwords = ['TOP SPAM WORDS']
    for item in od.items():
        count += 1
        if (count > length - 30):
            topspamwords.append(item[0])
    topspamdf = pd.DataFrame(topspamwords)
    combineddf = pd.concat([tophamdf,topspamdf],axis=1)
    return render_template("topspamham.html", data=combineddf.to_html())

@app.route('/modelperformance',methods=["POST","GET"])
def modelperformance():
    global cv,model1,accuracyscore,precision,recall
    session_var_value = session.get('filepath')
    df = pd.read_csv(session_var_value,encoding='latin-1')
    #print(df)
    X = df['text']
    y = df['class']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True,
                         stop_words='english')
    X_trains = cv.fit_transform(X_train)
    X_tests = cv.transform(X_test)


    if request.method == "POST":
        selectedalg = int(request.form['algorithm'])

        if (selectedalg == 1):
            model1 = SVC(kernel='linear')
            model1.fit(X_trains, y_train)
            y_pred = model1.predict(X_tests)
            accuracyscore = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,model="SVM with Kernel")
        if (selectedalg == 2):
            model2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
            model2.fit(X_trains, y_train)
            y_pred = model2.predict(X_tests)
            accuracyscore = accuracy_score(y_test, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore, model="Neural Network")

        if (selectedalg == 3):
            model3 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.075, max_features=2, max_depth=2, random_state=0)
            model3.fit(X_trains, y_train)
            y_pred = model3.predict(X_tests)
            accuracyscore = accuracy_score(y_test, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,model="Gradient Boost")

        if (selectedalg == 4):
            model4 = RandomForestClassifier(n_estimators=20)
            model4.fit(X_trains, y_train)
            y_pred = model4.predict(X_tests)
            accuracyscore = accuracy_score(y_test, y_pred)
            # accuracy_score = model.score(X_trains,y_trains)
            print(accuracyscore)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,model="RandomForest")

    return render_template('modelperformance.html')

@app.route('/prediction',methods=["POST","GET"])
def prediction():
    global accuracy,precision,recall
    if request.method == "POST":
        tweet = request.form['tweet']
        X_test_cv = cv.transform([tweet])
        predictiontrain = model1.predict(X_test_cv)
        print('Predictions: ', predictiontrain)
        return render_template('prediction.html',msg="success",model="SVM With Kernel",predictions=predictiontrain)
    return render_template('prediction.html')

@app.route('/bar_chart')
def bar():
        try:
            line_chart = pygal.Bar()
            line_chart.title = 'ACCURACY, PRECISION AND RECALL SCORES'
            line_chart.add('ACCURACY',accuracyscore )
            line_chart.add('PRECISION',precision )
            line_chart.add('RECALL', recall )
            graph_data = line_chart.render_data_uri()
            return render_template("bar_chart.html", graph_data=graph_data)
        except:
            return "OOPS! something went wrong"

if __name__ == '__main__':
    app.secret_key = ".."
    app.run()