from random import shuffle
#from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import MaxEntClassifier
from textblob import TextBlob
import re
import json


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


def replece_title(title):
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
    dict_level = {
                '2':['EVP','VP', 'Vice President'],
                '3':['Head of', 'Chief of', 'Director', 'Manager',],
                '4':['Team Lead', 'Senior', 'Specialist', 'Mid-Market', 'Sr.'],
                '5':['Junior', 'Representative','Assistant']}
    new_job = job.lower()
    finish_key = 5
    for key in dict_level.keys():
        for elem in dict_level[key]:
            if new_job.find(elem.lower()) >= 0:
                new_job = re.sub(elem.strip().lower(), '', new_job)
                if str(finish_key) > key:
                    finish_key = key
    return new_job.strip(), finish_key


def search_department(job, train):
    cl_depart = MaxEntClassifier(train)
    prob_dist = cl_depart.prob_classify(job)
    print(prob_dist.max())
    return prob_dist.max()


def search(title, train_departments):
    finder = search_in_hight_skill(title, dict_info)
    if finder is not None:
        finder.insert(0,title)
        return finder
    job, level = replace_job(title.strip())
    finder = search_department(job, train_departments)
    return [title,finder,level]


def machine_searching(main_title):

    main_title = replece_title(main_title)
    result = search(main_title, train_departments)
    return json.dumps({'Department': result[1], 'Level': result[2]}, indent=4)


dict_info = open_dict()
train_departments = create_set_department(dict_info)