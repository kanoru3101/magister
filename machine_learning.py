from random import shuffle
from textblob import classifiers
from textblob import TextBlob
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.linear_model import LogisticRegression,SGDClassifier
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier, SklearnClassifier, MaxentClassifier, DecisionTreeClassifier, accuracy
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from nltk.corpus import stopwords
import re
import json
import time


def open_dict():
    dict = json.load(open("dict.txt"))
    return dict


def create_set_department(dict_general):
    train_depart_set = []
    for key in dict_general.keys():
        for list in dict_general[key]:
            train_depart_set.append((list[0].lower().strip(), key))
    shuffle(train_depart_set)
    return train_depart_set


def create_featStructs(train):
    """
    create features for nltk classification
    :param train:
    :return:
    """
    tokenize_train = [(word_feats(item[0]), item[1]) for item in train] #item[0] - title job, item[1] - department
    return tokenize_train


def replece_title(title):
    """
    delete name company. Example "Software Developer Working Student at Soley" -> "Software Developer Working Student"
    :param title:
    :return:
    """
    if title.find(' at '):
        title = re.sub(r'((\sat\s).+)','', title)
    if title.find(' - '):
        title = re.sub(r'((\s–\s).+)','',title)
    return title.strip().lower()


def search_in_hight_skill(title,dict_general):
    text = TextBlob(title)
    text = text.words
    for word in text:
        for job in dict_general['Founder']:
            if job[0].strip().lower() == word:
                return ['Founder', '1']
        for job in dict_general['CEO']:
            if job[0].strip().lower() == word:
                return ['CEO', '2']
        for job in dict_general['Owner']:
            if job[0].strip().lower() == word:
                return ['Owner', '1']


def replace_job(job):
    """
    delete word as EVP, VP, Senior, Middle, Junior
    :param job:
    :return:
    """
    dict_level = {
                '2':['EVP','VP', 'Vice President'],
                '3':['Head of', 'Chief of', 'Director', 'Manager',],
                '4':['Team Lead', 'Senior', 'Specialist', 'Mid-Market', 'Sr.'],
                '5':['Junior', "Middle", 'Representative','Assistant']}
    new_job = job.lower()
    finish_key = 5
    for key in dict_level.keys():
        for elem in dict_level[key]:
            if new_job.find(elem.lower()) >= 0:
                new_job = re.sub(elem.strip().lower(), '', new_job)
                if str(finish_key) > key:
                    finish_key = key
    return new_job.strip()



def word_feats(words):
    """
    create feats for classifier. Example "I love this sandwich" -> {'I': True, 'love': True, 'sandwich.': True}
    :param words:
    :return:
    """
    stopset = list(set(stopwords.words('english')))
    return dict([(word, True) for word in words.split() if word not in stopset])





def searchNaiveBayesClassifier(title, train_departments):
    """
    NaiveBayesClassifier nltk
    :param title:
    :param train_departments:
    :return:
    """

    timeTraning = time.time()
    classifier = NaiveBayesClassifier.train(train_departments)
    timeTraning = time.time() - timeTraning

    test_sent_features = word_feats(title)

    timeClassify = time.time()
    found_department = classifier.classify(test_sent_features)
    timeClassify = time.time() - timeClassify

    probability = classifier.prob_classify(test_sent_features)
    print(probability.prob(found_department))

    return [found_department,
            probability.prob(found_department),
            accuracy(classifier, train_departments[1000:]),
            timeClassify,
            timeTraning,
            ]

def searchMultinomialNB(title, train_departments):
    """

    :param title:
    :param train_departments:
    :return:
    """

    timeTraning = time.time()
    classifier = SklearnClassifier(MultinomialNB())
    classifier.train(train_departments)
    timeTraning = time.time() - timeTraning

    test_sent_features = word_feats(title)

    timeClassify = time.time()
    found_department = classifier.classify(test_sent_features)
    timeClassify = time.time() - timeClassify

    probability = classifier.prob_classify(test_sent_features)
    print(probability.prob(found_department))

    return [found_department,
            probability.prob(found_department),
            accuracy(classifier, train_departments[1000:]),
            timeClassify,
            timeTraning,
            ]


def searchBernoulliNB(title, train_departments):
    """

    :param title:
    :param train_departments:
    :return:
    """
    timeTraning = time.time()
    classifier = SklearnClassifier(BernoulliNB())
    classifier.train(train_departments)
    timeTraning = time.time() - timeTraning

    test_sent_features = word_feats(title)

    timeClassify = time.time()
    found_department = classifier.classify(test_sent_features)
    timeClassify = time.time() - timeClassify

    probability = classifier.prob_classify(test_sent_features)
    print(probability.prob(found_department))

    return [found_department,
            probability.prob(found_department),
            accuracy(classifier, train_departments[1000:]),
            timeClassify,
            timeTraning,
            ]


def searchSVCClassifier(title, train_departments):
    """
    SVC Classifier
    :param title:
    :param train_departments:
    :return:
    """
    timeTraning = time.time()
    classifier = SklearnClassifier(SVC(probability=True))
    classifier.train(train_departments)
    timeTraning = time.time() - timeTraning

    test_sent_features = word_feats(title)

    timeClassify = time.time()
    found_department = classifier.classify(test_sent_features)
    timeClassify = time.time() - timeClassify

    probability = classifier.prob_classify(test_sent_features)
    print(probability.prob(found_department))

    return [found_department,
            probability.prob(found_department),
            accuracy(classifier, train_departments[1000:]),
            timeClassify,
            timeTraning,
            ]


def searchLinearSVC(title, train_departments):
    """
    Linear SVC
    :param title:
    :param train_departments:
    :return:
    """
    timeTraning = time.time()
    #classifier = SklearnClassifier(LinearSVC(probability=True))
    classifier = SklearnClassifier(SVC(kernel='linear', probability=True))
    classifier.train(train_departments)
    timeTraning = time.time() - timeTraning

    test_sent_features = word_feats(title)

    timeClassify = time.time()
    found_department = classifier.classify(test_sent_features)
    timeClassify = time.time() - timeClassify

    probability = classifier.prob_classify(test_sent_features)
    print(probability.prob(found_department))

    return [found_department,
            probability.prob(found_department),
            accuracy(classifier, train_departments[1000:]),
            timeClassify,
            timeTraning,
            ]


def searchMaxentClassifier(title, train_departments):
    """

    :param title:
    :param train_departments:
    :return:
    """
    timeTraning = time.time()
    classifier = MaxentClassifier.train(train_departments, max_iter=5)
    timeTraning = time.time() - timeTraning

    test_sent_features = word_feats(title)

    timeClassify = time.time()
    found_department = classifier.classify(test_sent_features)
    timeClassify = time.time() - timeClassify

    probability = classifier.prob_classify(test_sent_features)
    print(probability.prob(found_department))

    return [found_department,
            probability.prob(found_department),
            accuracy(classifier, train_departments[1000:]),
            timeClassify,
            timeTraning,
            ]


def searchDecisionTreeClassifier(title, train_departments):
    """

    :param title:
    :param train_departments:
    :return:
    """
    timeTraning = time.time()
    classifier = DecisionTreeClassifier.train(train_departments)
    timeTraning = time.time() - timeTraning

    test_sent_features = word_feats(title)

    timeClassify = time.time()
    found_department = classifier.classify(test_sent_features)
    timeClassify = time.time() - timeClassify

    #probability = classifier.prob_classify_many(test_sent_features)
    #print(probability.prob(found_department))

    return [found_department,
            #probability.prob(found_department),
            0,
            accuracy(classifier, train_departments[1000:]),
            timeClassify,
            timeTraning,
            ]


def searchLogisticRegressionClassifier(title, train_departments):
    """

    :param title:
    :param train_departments:
    :return:
    """
    timeTraning = time.time()
    classifier = SklearnClassifier(LogisticRegression())
    classifier.train(train_departments)
    timeTraning = time.time() - timeTraning

    test_sent_features = word_feats(title)

    timeClassify = time.time()
    found_department = classifier.classify(test_sent_features)
    timeClassify = time.time() - timeClassify

    probability = classifier.prob_classify(test_sent_features)
    print(probability.prob(found_department))

    return [found_department,
            probability.prob(found_department),
            accuracy(classifier, train_departments[1000:]),
            timeClassify,
            timeTraning,
            ]


def searchSGDClassifier_classifier(title, train_departments):
    """

    :param title:
    :param train_departments:
    :return:
    """
    timeTraning = time.time()
    classifier = SklearnClassifier(SGDClassifier(loss='log'))
    classifier.train(train_departments)
    timeTraning = time.time() - timeTraning

    test_sent_features = word_feats(title)

    timeClassify = time.time()
    found_department = classifier.classify(test_sent_features)
    timeClassify = time.time() - timeClassify

    probability = classifier.prob_classify(test_sent_features)
    print(probability.prob(found_department))

    return [found_department,
            probability.prob(found_department),
            accuracy(classifier, train_departments[1000:]),
            timeClassify,
            timeTraning,
            ]




#don`t work clasiffication
def searchNuSVC_classifier(title, train_departments):
    """
    Nu-Support Vector Classification.
    :param title:
    :param train_departments:
    :return:
    """
    classifier = SklearnClassifier(NuSVC())
    classifier.train(train_departments)
    test_sent_features = word_feats(title)
    return classifier.classify(test_sent_features)





def search(title, train_departments):
    """
    :param title:
    :param train_departments:
    :return:
    """
    title = replace_job(title.strip())
    finder = search_department(title, train_departments)
    return [title,finder]


def machine_searching(main_title):
    """

    :param main_title:
    :return:
    """

    main_title = replece_title(main_title)
    main_title = replace_job(main_title.strip())

    timeNaiveBayes = time.time()
    nltkNaiveBayesClassifier = searchNaiveBayesClassifier(main_title, train_depart_for_nltk)
    timeNaiveBayes = time.time() - timeNaiveBayes

    timeMultinomialNB = time.time()
    MultinomialNB = searchMultinomialNB(main_title, train_depart_for_nltk)
    timeMultinomialNB = time.time() - timeMultinomialNB

    timeBernoulliNB = time.time()
    BernoulliNB = searchBernoulliNB(main_title, train_depart_for_nltk)
    timeBernoulliNB = time.time() - timeBernoulliNB

    timeSVC = time.time()
    svcClassifier = searchSVCClassifier(main_title, train_depart_for_nltk)
    timeSVC = time.time() - timeSVC

    timeLinearSVC = time.time()
    nltkLinearSVC = searchLinearSVC(main_title,train_depart_for_nltk)
    timeLinearSVC = time.time() - timeLinearSVC

    timeMaxent = time.time()
    nltkMaxentClassifier = searchMaxentClassifier(main_title,train_depart_for_nltk)
    timeMaxent = time.time() - timeMaxent

    timeDecisionTree = time.time()
    nltkDecisionTreeClassifier = searchDecisionTreeClassifier(main_title,train_depart_for_nltk)
    timeDecisionTree = time.time() - timeDecisionTree

    timeLogisticRegression = time.time()
    LogisticRegression_classifier = searchLogisticRegressionClassifier(main_title,train_depart_for_nltk)
    timeLogisticRegression = time.time() - timeLogisticRegression

    timeSGD = time.time()
    SGDClassifier_classifier = searchSGDClassifier_classifier(main_title,train_depart_for_nltk)
    timeSGD = time.time() - timeSGD


    #nuSVC_classifier = searchNuSVC_classifier(main_title,train_depart_for_nltk)

    return json.dumps({'Title:': main_title,
                       "NaiveBayesClassifier": {
                           "Result": nltkNaiveBayesClassifier[0],
                           "Acuracy result": nltkNaiveBayesClassifier[1],
                           "Acuracy dataset": nltkNaiveBayesClassifier[2],
                           "Time classify": nltkNaiveBayesClassifier[3],
                           "Time train": nltkNaiveBayesClassifier[4],
                           "Time total": timeNaiveBayes
                        },
                       "MultinomialNB": {
                           "Result": MultinomialNB[0],
                           "Acuracy result": MultinomialNB[1],
                           "Acuracy dataset": MultinomialNB[2],
                           "Time classify": MultinomialNB[3],
                           "Time train": MultinomialNB[4],
                           "Time total": timeMultinomialNB
                       },
                       "BernoulliNB": {
                           "Result": BernoulliNB[0],
                           "Acuracy result": BernoulliNB[1],
                           "Acuracy dataset": BernoulliNB[2],
                           "Time classify": BernoulliNB[3],
                           "Time train": BernoulliNB[4],
                           "Time total": timeBernoulliNB
                       },
                       "SVC": {
                           "Result": svcClassifier[0],
                           "Acuracy result": svcClassifier[1],
                           "Acuracy dataset": svcClassifier[2],
                           "Time classify": svcClassifier[3],
                           "Time train": svcClassifier[4],
                           "Time total": timeSVC
                       },
                       "LinearSVC": {
                           "Result": nltkLinearSVC[0],
                           "Acuracy result": nltkLinearSVC[1],
                           "Acuracy dataset": nltkLinearSVC[2],
                           "Time classify": nltkLinearSVC[3],
                           "Time train": nltkLinearSVC[4],
                           "Time total": timeLinearSVC
                       },
                       #"nuSVC_classifier": nuSVC_classifier,
                       "MaxentClassifier": {
                           "Result": nltkMaxentClassifier[0],
                           "Acuracy result": nltkMaxentClassifier[1],
                           "Acuracy dataset": nltkMaxentClassifier[2],
                           "Time classify": nltkMaxentClassifier[3],
                           "Time train": nltkMaxentClassifier[4],
                           "Time total": timeMaxent
                       },
                       "DecisionTreeClassifie": {
                           "Result": nltkDecisionTreeClassifier[0],
                           "Acuracy result": nltkDecisionTreeClassifier[1],
                           "Acuracy dataset": nltkDecisionTreeClassifier[2],
                           "Time classify": nltkDecisionTreeClassifier[3],
                           "Time train": nltkDecisionTreeClassifier[4],
                           "Time total": nltkDecisionTreeClassifier
                       },
                       "LogisticRegression": {
                           "Result": LogisticRegression_classifier[0],
                           "Acuracy result": LogisticRegression_classifier[1],
                           "Acuracy dataset": LogisticRegression_classifier[2],
                           "Time classify": LogisticRegression_classifier[3],
                           "Time train": LogisticRegression_classifier[4],
                           "Time total": timeLogisticRegression
                       },
                       "SGDClassifier": {
                           "Result": SGDClassifier_classifier[0],
                           "Acuracy result": SGDClassifier_classifier[1],
                           "Acuracy dataset": SGDClassifier_classifier[2],
                           "Time classify": SGDClassifier_classifier[3],
                           "Time train": SGDClassifier_classifier[4],
                           "Time total": timeSGD
                       },
                       }, indent=4)


dict_info = open_dict()
train_departments = create_set_department(dict_info)
train_depart_for_nltk = create_featStructs(train_departments)  # update train set for nltk classify