import json
import csv
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def read_csv():
    """
    read csv file and create list
    :return: list
    """
    some_list_in_csv = []
    with open('dataset.csv', 'r', newline=""    ) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            some_list_in_csv.append(row)
    return some_list_in_csv


def create_dict_department(list_csv):
    """
    create dictionary with information about departments and levels
    :param list_csv:
    :return: dictionary{'Department': [job, level]..}
    """
    names_department = list_csv[1]  # [0] - name jobs [1] - space
    dict_departments = {}
    for department in names_department[2:16]:
        dict_departments[department] = []
    list_worker = list_csv[2:]
    for worker in list_worker:
        for i in range(2,16):
            if worker[i] != '':
                dict_departments[names_department[i]].append([worker[0].lower(), str(worker[i])])
    return dict_departments




def stopWordsAndTokenize(sentence):
    """
    delete words as: for, a, to, the...
    :param sentence: job title
    """
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(sentence)
    wordsFiltered = ''
    for w in words:
        if w not in stopWords:
            wordsFiltered += ' ' + w
    return wordsFiltered.strip()


def replece_title(title):
    """
    removes the level of employee
    :return: job title
    """
    if title.find(' at '):
        new_title = re.sub(r'((\sat\s).+)','', title)
    if title.find(' - '):
        new_title = re.sub(r'((\s–\s).+)','',title)
    dict_level = {'2': ['EVP', 'VP', 'Vice President'],
                  '3': ['Head of', 'Chief of', 'Director', 'Manager'],
                  '4': ['Team Lead', 'Senior', 'Specialist'],
                  '5': ['Junior', 'Representative', 'Assistant']}
    for key in dict_level:
        for elem in dict_level[key]:
            if title.find(elem.lower()) >= 0:
                new_title = re.sub(elem.strip().lower(), '', title)
                return new_title.strip()
    return title.strip().lower()


def save_dict(dict):
    """
    save dictionary
    """
    json.dump(dict, open("dict.txt", 'w'))
    print('New Dict saved')


list_csv = read_csv()
dict_general = create_dict_department(list_csv)

for key in dict_general.keys():
    for list_info in dict_general[key]:
        list_info[0] = replece_title(list_info[0])
        list_info[0] = stopWordsAndTokenize(list_info[0])

save_dict(dict_general)

print('Size departments')
for key in dict_general.keys():
    print(key,len(dict_general[key]))
