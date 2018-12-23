import machine_learning
import maxEntClassifier
import fd


#result = machine_learning.machine_searching('partner')
#print(result)
job = None
while job != "exit":
    job = input()
    result = machine_learning.machine_searching(job)
    #print("NaiveBayesClassifier: " + result)
    #resultMEC = maxEntClassifier.machine_searching(job)
   # print("MaxEntClassifier: " +  resultMEC)
    #result = fd.machine_searching(job)
    print(result)