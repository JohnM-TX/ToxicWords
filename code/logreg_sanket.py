# Sanket Keni 
'''
CV score for class toxic is 0.9758815956729977
CV score for class severe_toxic is 0.9885067270242905
CV score for class obscene is 0.9919493883065732
CV score for class threat is 0.9866684407022007
CV score for class insult is 0.9806593278329583
CV score for class identity_hate is 0.981040742648163
Total CV score is 0.9841177036978639
Public LB - 0.9787
'''
import numpy as np
import pandas as pd
import nltk
# from playsound import playsound
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, cross_val_predict
from scipy.sparse import hstack
import timeit
import re
stemmer = nltk.stem.snowball.SnowballStemmer('english')

############### send notification on smartphone
# from urllib.parse import urlencode
# from urllib.request import Request, urlopen
# url = 'https://www.pushsafer.com/api' # Set destination URL here
# post_fields = {                       # Set POST fields here
# 	"t" : "Python code execution complete",
# 	"m" : "task finished" + str(k),
# 	"d" : "a",
# 	"u" : url,
# 	"k" : "*************"
# 	}
# def notify():
#     request = Request(url, urlencode(post_fields).encode())
#     json = urlopen(request).read().decode()
#     print(json)


# notify when code has completed execution
# def audio():
#     playsound('C:\\Users\\Sanket Keni\\Music\\notification.mp3')


train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
'''
train['comment_text']=train['comment_text'].apply(lambda x :clean(x))
test['comment_text']=test['comment_text'].apply(lambda x :clean(x))
'''


def cleaned(comment):
    comment=comment.lower()
    comment=re.sub("\\n"," ",comment)
    comment=re.sub("\d{1,}","",comment)
    comment=re.sub("\.{1,}",".",comment)
    comment=re.sub("\:{1,}","",comment)
    comment=re.sub("\;|\=|\%|\^|\_"," ",comment)
    comment=re.sub("\""," ",comment)
    comment=re.sub("\'{2,}","",comment)
    comment=re.sub("\/|\!"," ",comment)
    comment=re.sub("\?"," ",comment)
    comment=re.sub("\#"," ",comment)
    comment=re.sub("\,|\@|\|"," ",comment)
    comment=re.sub("\(|\)"," ",comment)
    comment=re.sub("\S+jpg"," ",comment)
    comment=re.sub("\S*wikip\S+","",comment)               
    comment=re.sub("\[.*?\]"," ",comment)
    comment=re.sub("\-"," ",comment)
    '''comment=re.sub("\"|:|@|,|\/|\=|;|\.|\'|\?|\!|\||\+|\~|\-|\#"," ",comment)
    comment=re.sub("\.{1,}",".",comment)
    comment=re.sub("\[.*?\]","",comment)
    comment=re.sub("www\S+","",comment)
    comment=re.sub("\_"," ",comment)
    comment=re.sub("http","",comment)'''
    comment=re.sub(r'[^\x00-\x7F]+',' ', comment) # remove non ascii
    comment=re.sub("\s+"," ",comment)
    comment = ' '.join( [w for w in comment.split() if len(w)>1])
    comment = ' '.join( [stemmer.stem(w) for w in comment.split()])
    comment = comment.strip()
    return comment

print('cleaning')
train['comment_text']=train['comment_text'].apply(lambda x :cleaned(x))
test['comment_text']=test['comment_text'].apply(lambda x :cleaned(x))
# audio()

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])
'''
stopwords = nltk.corpus.stopwords.words('english')
mystopwords = "aa abc"
'''

print('words')
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=5000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
#audio()

print('chars')
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 3),
    max_features=5000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
# audio()

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
predfile = pd.DataFrame.from_dict({'id': train['id']})

for class_name in class_names:
    if (class_name in ['toxic']):
        train_target = train[class_name]
        classifier = LogisticRegression(C=0.63, solver='sag', class_weight = "balanced") # sag arge datasets and bivariate
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        cv_preds = cross_val_predict(classifier, train_features, train_target, cv=3, method='predict_proba')
        predfile[class_name] = cv_preds[:, 1]
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))
        classifier.fit(train_features, train_target)
        submission[class_name] = classifier.predict_proba(test_features)[:, 1]
    
    elif(class_name in ["severe_toxic", "insult"]):
        train_target = train[class_name]
        classifier = LogisticRegression(C=0.38, solver='sag') # sag large datasets and bivariate
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        cv_preds = cross_val_predict(classifier, train_features, train_target, cv=3, method='predict_proba')
        predfile[class_name] = cv_preds[:, 1]
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))
        classifier.fit(train_features, train_target)
        submission[class_name] = classifier.predict_proba(test_features)[:, 1]
        
    elif(class_name in ["threat", "identity_hate"]):
        train_target = train[class_name]
        classifier = LogisticRegression(C=0.45, solver='sag') # sag large datasets and bivariate
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        cv_preds = cross_val_predict(classifier, train_features, train_target, cv=3, method='predict_proba')
        predfile[class_name] = cv_preds[:, 1]
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))
        classifier.fit(train_features, train_target)
        submission[class_name] = classifier.predict_proba(test_features)[:, 1]
        
    elif(class_name == "obscene"):
        train_target = train[class_name]
        classifier = Ridge(alpha=20, solver='auto',max_iter=100, random_state=22,  tol=0.0005)
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        cv_preds = cross_val_predict(classifier, train_features, train_target, cv=3)
        predfile[class_name] = cv_preds
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))
        classifier.fit(train_features, train_target)
        submission[class_name] = classifier.predict(test_features)

print('Total CV score is {}'.format(np.mean(scores)))
# audio()
#notify()
predfile.to_csv('../ensembles/preds_logreg_sanket.csv', index=False)
submission.to_csv('../ensembles/test_logreg_sanket.csv', index=False)