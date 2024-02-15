import sys
from allTokens import *
import os
import ray
import time
import pandas as pd
from langdetect import detect

outputFilePath = "CleanedData/"


def dataLoad(dfName):
    try:
        if ".csv" in dfName:
            return pd.read_csv (dfName)
        elif ".feather" in dfName:
            return pd.read_feather(dfName)
        else:
            print(dfName+" database not correct")
            sys.exit(1)
    except Exception as e:
        print(e)
        sys.exit(1)

        
def removeHashTags(text):
    div = text.split("#")
    endExists = True
    i = len(div)-1
    while i>=0 and endExists:
        if len(div[i].strip().split(" "))  == 1:
            div.pop(i)
            i-=1
        else:
            endExists = False
        
    return " ".join(div).strip()


def removeRT(text):
    if text[0:2] == 'RT':
        return ":".join(text.split(":")[1:]).strip()
    return text  

def isEnglish(s):
    try:
        return(detect(s) == 'en')
    except:
        return('')

def cleanTweet(text):
    text = text.strip()
    for key,value in abbr_dict.items():
        text = re.sub(key,value,text)
    text = emoji_pattern.sub(r' ', text)
    text = removeHashTags(text)
    text = removeRT(text)
    text = re.sub(' +', ' ', text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub("@",'',text).lower()
    text = ' '.join(text.replace('\r', ' ').replace('\n', ' ').split())
    if not isEnglish(text):
        return ''
    return text.strip()


@ray.remote
def f(dft):
    dft['cleanText'] = dft.full_tweet_text.apply(cleanTweet)
    return dft[dft['cleanText']!='']

def createNewOutputFolder():
    if os.path.isfile(outputFilePath):
        for f in os.listdir(outputFilePath):
            os.remove(os.path.join(outputFilePath, f))
    else:
        os.mkdir(outputFilePath)

if __name__ == "__main__":
    if(len(sys.argv) <2):
        print("use python DataCleaning.py <csv or feather name>")
        sys.exit(1)
    df = dataLoad(sys.argv[1])[['tweet_hash', 'full_tweet_text']]
    print("data read")
    processCount = 50
    ray.init(num_cpus=processCount)
    finalRange = [i for i in range(0,len(df)+1,len(df)//processCount)]

    t = time.time()
    result_ids = []

    for i in range(len(finalRange)-1):
        result_ids.append(f.remote(df.iloc[finalRange[i]:finalRange[i+1]]))
    results = ray.get(result_ids)
    print(time.time()-t)
    createNewOutputFolder()
    
    for i in range(len(results)):
        print(i)
        results[i].reset_index().to_feather(outputFilePath+str(i)+".feather")
    del results,df,result_ids
    ray.shutdown()
