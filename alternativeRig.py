from __future__ import division
import sys
import random
import math

from cart import *
from models import *
from searchers import *
from options import *
import time
from sk import *
import sys
sys.path.insert(0, 'defectprediction')
from main import *


sys.dont_write_bytecode = True
#Dr.M
rand=  random.random # generate nums 0..1
any=   random.choice # pull any from list
sqrt= math.sqrt  #square root function

def display(modelName,searcher,runTimes,scores,historyhi=[],historylo=[]):
  assert(len(runTimes) == len(scores)),'Ouch! it hurts'
  print "==============================================================="
  print "Model Name: %s"%modelName
  print "Searcher Name: %s"%searcher
  print "Options Used: ",
  print myoptions[searcher]
  import time
  print ("Date: %s"%time.strftime("%d/%m/%Y"))
  print "Average running time: %f " %mean(runTimes)
  if(len(historyhi)!=0):
    for x in xrange(myModelobjf[modelName]):
      print "Objective No. %d: High: %f Low: %f"%(x+1,historyhi[x],historylo[x])
  #for i in range(0,len(runTimes)):
  #  print "RunNo: %s RunTime: %s Score: %s"%(i+1,runTimes[i],scores[i])
  #print scores
  print xtile(scores,width=25,show=" %1.2f")
  print "==============================================================="

def getdata():
	from os import walk
	returnList = []
	dirs = []
	for (dirpath, dirnames, filenames) in walk("./defectprediction/data"):
	    dirs.extend(dirnames)
	    break
	#print dirs
	for dir in dirs:
		path = "./defectprediction/data/" + str(dir)
		#print path
		import os
		templst = [path +"/" + x for x in sorted(os.listdir(path))]
		assert(len(templst) >= 3 ),"Something's Wrong"
		#print =
		training = templst[0]
		tuning = templst[1]
		testing = templst[2]
		#print training,tuning,testing
		returnList.append([training,tuning,testing])


	return returnList

def stats(listl):
  def median(lst,ordered=False):
    if not ordered: lst= sorted(lst)
    n = len(lst)
    p = n//2
    if n % 2: return lst[p]
    q = p - 1
    q = max(0,min(q,n))
    return (lst[p] + lst[q])/2
  from scipy.stats import scoreatpercentile
  q1 = scoreatpercentile(listl,25)
  q3 = scoreatpercentile(listl,75)  
  #print "IQR : %f"%(q3-q1)
  #print "Median: %f"%median(listl)
  return median(listl),(q3-q1)

  

def multipleRun():
   from collections import defaultdict
   r = 1
   for f in getdata():
	   train = f[0]
	   predict = f[1]
	   test = f[2]
	   
	   for klass in [TunedWhere]:#TunedCart]:#,DTLZ5,DTLZ6,DTLZ7]:
	     print "Model Name: %s"%klass.__name__
	     evalScores=defaultdict(list)
	     print ("Date: %s"%time.strftime("%d/%m/%Y"))
	     bmin = -3.2801
	     bmax = 5.6677
	     naiveWhere = []
	     print "========================TunedWhere================================"
	     print "Training : ",train
	     print "Predict: ",predict
	     print "Test: ",test
	     tstart = time.time()
	     print "Start Time: ",tstart
	     for searcher in [DE,Seive2_Initial]:
	       print "Searcher: ",searcher.__name__
	       n = 0.0
	       The.option.baseLine = False
	       The.option.tuning  = True
	       listTimeTaken = []
	       listScores = []
	       list_eval = []
	       random.seed(6)
	       historyhi=[-9e10 for count in xrange(myModelobjf[klass.__name__])]
	       historylo=[9e10 for count in xrange(myModelobjf[klass.__name__])]
	       print searcher.__name__,
	       for _ in range(r):
	         search = searcher(klass(train,predict),"display2",bmin,bmax)
	         #search = searcher(klass(),"display2",bmin,bmax)
	         print ".", 
	         solution,model = search.evaluate()

	       print "Time for tuning: ", time.time() - tstart
	       print "Number of Evaluation: ",model.no_eval
	       print "High Score: ",solution,scores
	       tstart = time.time()

	       tr = f[0]
	       ts = f[2]
	       print "Tuned Parameters: ",solution.dec
	       temp_scores = [runPredict(solution.dec,tr,ts) for x in xrange(10)]
	       evalScores[klass.__name__] = temp_scores
	       median,iqr = stats(temp_scores)
	       print "Median: ",median," IQR: ",iqr
	       print "Time for Running: ",time.time() - tstart

	   print "==========================NaiveWhere=============================="
	   #print "Baseline Finished: ",bmin,bmax
	   print "Training : ",train
	   print "Test: ",test
	   tstart = time.time()
	   print "Start Time: ",
	   temp_scores = [NaiveWhere(train,test) for _ in xrange(10)]
	   evalScores['NaiveWhere'] = temp_scores
	   median,iqr = stats(temp_scores)
	   print "Median: ",median," IQR: ",iqr
	   print "Time for Experiment: ",time.time() - tstart
	   print evalScores
	   callrdivdemo(evalScores,"%5.0f")


def step2():
    rdivDemo([
      ["Romantic",385,214,371,627,579],
      ["Action",480,566,365,432,503],
      ["Fantasy",324,604,326,227,268],
      ["Mythology",377,288,560,368,320]])   


def callrdivdemo(eraCollector,show="%5.2f"):
  #print eraCollector
  #print "callrdivdemo %d"%len(eraCollector.keys())
  keylist = eraCollector.keys() 
  #print keylist
  variant = len(keylist)
  #print variant
  rdivarray=[]
  for y in xrange(variant):
      temp = eraCollector[keylist[y]]
      temp.insert(0,str(keylist[y]))
      rdivarray.append(temp)
  rdivDemo(rdivarray,show) 

if __name__ == '__main__': 
 multipleRun()
 #step2()



