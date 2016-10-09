"""
This code is to generate csv by using random function which generates value from [0,1)
"""
import csv
import random

def readFile(fileName):
    try:
        lines = [line.rstrip('\r\n') for line in open(fileName)]
        return lines
    except Exception, e:
        print " Error occurred when opening file ", fileName, " error stack = ", e

def getPrediction(item, questionsData, userData):
    #print " Received request with question data set ", len(questionsData), " user dataset", len(userData), item
    qsId = item[0]
    userId = item[1]

    questionTag = list(map(lambda x: x[1] ,list(filter(lambda x: x[0] == qsId , questionsData))))
    userTag = list(map(lambda x: x[1].split('/') ,list(filter(lambda x: x[0] == userId , userData))))
    #print "Running for qsId = ", qsId, " userId = " , userId, "filter questionTag = ", questionTag, " filter userTag =", userTag

    return 1 if questionTag in userTag else 0

if __name__ == "__main__":

    try :
        lines = readFile('validate_nolabel.txt')
        lines.pop(0) #Remove the header line
        lines = [line.split(',') for line in lines]

        questionsData = readFile('question_info.txt');
        questionsData = [ questionItem.split() for questionItem in questionsData ]
        print len(questionsData)

        userData = readFile('user_info.txt');
        userData = [ userItem.split() for userItem in userData ]
        print len(userData)

        f = open('result.csv', 'w')
        fileWriter = csv.writer(f)
        fileWriter.writerow(["qid","uid","label"])
        count = 0
        for item in lines:
            if (count in [1000, 2000, 5000,7000, 10000, 15000, 20000, 25000, 30000]):
                print "......Processed total ", count, " lines......."
            #val = getPrediction(item, questionsData, userData)
            val = random.random()
            #print val
            fileWriter.writerow([item[0], item[1], str(val)])
            count = count + 1
        print count

    #except Exception, e:
    #    print " Something went wrong in main function ", e

    finally:
        f.close()