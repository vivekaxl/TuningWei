from __future__ import division
import sys
import random
import math

from cart import *
from randomforest import *
from witschey_main import *
#import numpy as np
from models import *
from searchers import *
from options import *
from utilities import *
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
		#print templst
		training = templst[:-2]
		tuning = templst[-2]
		testing = templst[-1]
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
	   
	   # for klass in [TunedCart]:#,DTLZ5,DTLZ6,DTLZ7]:
	   #   print "Model Name: %s"%klass.__name__
	   #   eraCollector=defaultdict(list)
	   #   timeCollector=defaultdict(list)
	   #   evalCollector=defaultdict(list)
	   #   print ("Date: %s"%time.strftime("%d/%m/%Y"))
	   #   #bmin,bmax = tempC.baseline(tempC.minR, tempC.maxR) 
	   #   bmin = -3.2801
	   #   bmax = 5.6677
	   #   naiveWhere = []
	   #   print "========================TunedWhere================================"
	   #   #print "Baseline Finished: ",bmin,bmax
	   #   print "Training : ",train
	   #   print "Predict: ",predict
	   #   print "Test: ",test
	   #   tstart = time.time()
	   #   print "Start Time: ",tstart
	   #   for searcher in [Seive2_Initial]:
	   #     n = 0.0
	   #     The.option.baseLine = False
	   #     The.option.tuning  = True
	   #     listTimeTaken = []
	   #     listScores = []
	   #     list_eval = []
	   #     random.seed(6)
	   #     historyhi=[-9e10 for count in xrange(myModelobjf[klass.__name__])]
	   #     historylo=[9e10 for count in xrange(myModelobjf[klass.__name__])]
	   #     print searcher.__name__,
	   #     for _ in range(r):
	   #       test = searcher(klass(train,predict),"display2",bmin,bmax)
	   #       print ".", 
	        
	   #       t1 = time.time()
	   #       solution,score,model = test.evaluate()

	   #       for x in xrange(model.objf):
	   #         #print len(model.past[x].listing)
	   #         #print x
	   #         historyhi[x]=max(model.past[x].historyhi,historyhi[x])
	   #         historylo[x]=min(model.past[x].historylo,historylo[x])
	   #         sys.stdout.flush()
	   #       timeTaken = (time.time() - t1) * 1000
	   #       #listTimeTaken.append(timeTaken)
	   #       list_eval.append(model.no_eval)
	   #       listScores.append(score)
	   #       timeCollector[searcher.__name__]=listTimeTaken
	   #     eraCollector[searcher.__name__]=listScores
	   #     evalCollector[searcher.__name__]=list_eval

	   #     tr = f[0]
	   #     ts = f[2]
	   #     print "Tuned Parameters: ",solution
	   #     The.option.baseLine = True
	   #     The.option.tuning  = False
	   #     #runPredict(solution,tr,ts)
	   #     median,iqr = stats([runCart(solution,tr,ts) for x in xrange(10)])
	   #     print "Median: ",median," IQR: ",iqr
	      
	   #      #print "Score: %f"%(score)
	   #     print
	   #     print "Time for Experiment: ",time.time() - tstart

	   print "==========================NaiveWhere=============================="
	   #print "Baseline Finished: ",bmin,bmax
	   print "Training : ",train
	   print "Predict: ",predict
	   print "Test: ",test
	   tstart = time.time()
	   print "Start Time: ",tstart
	   #NaiveWhere(f[0],f[2])

	   median,iqr = stats([NaiveRF(f[0],f[2]) for x in xrange(10)])
	   print "Median: ",median," IQR: ",iqr
	   print "Time for Experiment: ",time.time() - tstart


     
     # listbaseline = []
     # for _ in range(r):
     #   testB = Baseline(klass(),"display2",bmin,bmax)
     #   tmp = testB.evaluate()
     #   listbaseline.extend(tmp)
     # print "Baseline: length is: ",len(listbaseline)
     # eraCollector['baseline'] = listbaseline
     #callrdivdemo(eraCollector)
     #raise Exception("I know python!")
     #print eraCollector
     #print evalCollector
     #print timeCollector
     # print "=========================================================="
     # callrdivdemo(eraCollector)
     # callrdivdemo(evalCollector,"%5.0f")
     #callrdivdemo(timeCollector)


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
      #print "Length of array: %f"%len(eraCollector[keylist[y]][x])
      temp = eraCollector[keylist[y]]
      #print temp
      temp.insert(0,str(keylist[y]))
      #print temp
      rdivarray.append(temp)
  rdivDemo(rdivarray,show) 
  


def testDE():
  for klass in [Viennet]:
    random.seed(6)
    test = DE(klass(),"display2")          
    print test.evaluate()

if __name__ == '__main__': 
 # random.seed(1)
 # nums = [random.random()**2 for _ in range(100)]
 # print xtile(nums,lo=0,hi=1.0,width=25,show=" %3.2f")
 # model = ZDT1()
 # model.testgx()
 # for klass in [ZDT1]:
 #   print klass.__name__

 #for x in xrange(1,8):
 #  print "========================================="
 #  opt = 1000*x
 #  myoptions['Seive']['initialpoints'] = str(opt)
 #  multipleRun()

 multipleRun()
 # t = getdata()
 # for f in t:
 # 	print "Training: ",f[0]
 # 	print "Tuning: ",f[1]
 # 	print "Testing: ",f[2]
 	# print
 #testDE()
 #part6()
 #step2()



