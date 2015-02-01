from __future__ import division
import sys
from os import listdir
from table import *
from os.path import isfile, join
from settings import *
from cart_de import *
from sk import *
from time import strftime
from rf_cart_prediction import *

def myrdiv(d):
	def pre(dist):
		l=dist.items()[0][-1]
		k = dist.items()[0][0]
		return [k].extend(l)
	stat = []
	for key,val in d.iteritems():
		val.insert(0,key)
		stat.append(val)
	return stat

def createfile():
	f = open('myresult','w').close()

def writefile(s):
	f = open('myresult', 'a')
	f.write(s+'\n')
	f.close()


def cart_file(path="./data"):
  def saveSat(score,dataname):
    class1 = dataname +": N-Def"
    class2 = dataname +": Y-Def"
    name = [pd,pf,prec,g]
    for i, s in enumerate(name):
      s[class1]= s.get(class1,[])+[float(score[0][i]/100)]
      s[class2]= s.get(class2,[])+[float(score[1][i]/100)]

  def printresult(dataset):
    print "\n" + "+" * 20 + "\n DataSet: "+dataset + "\n" + "+" * 20
    for i, k in enumerate(["pd", "pf","prec","g"]):
        # pdb.set_trace()
        express = "\n"+"*"*10+k+"*"*10
        print express
        writefile(express)
        rdivDemo(myrdiv(lst[i]))
    writefile("End time :" +strftime("%Y-%m-%d %H:%M:%S"))
    writefile("\n"*2)
    print "\n"

  def predicttest(predict, testname):
    for i in xrange(10):
      The.data.predict = predict
      score = main()
      saveSat(score, testname)

  def cart_tunetest(predict):
    The.classifier.tuned = True
    cart_de()
    The.option.tuning = False
    predicttest(predict,"Tuned")
    The.classifier.tuned = False


  def cart_basetest(predict):
    The.classifier.cart = True
    The.cart.criterion = "entropy"
    The.cart.max_features = None
    The.cart.max_depth = None
    The.cart.min_samples_split = 2
    The.cart.min_samples_leaf = 1
    The.cart.max_leaf_nodes = None
    The.cart.random_state = 0
    predicttest(predict, "Cart")
    The.classifier.cart = False


  def rndfsttest(predict):
    The.classifier.rdfor = True
    predicttest(predict, "Rdfor")
    The.classifier.rdfor = False

  random.seed(10)
  createfile()
  folders = [f for f in listdir(path) if not isfile(join(path, f))]
  for one in folders[5:7]:
    pd, pf, prec, g = {},{},{},{}
    lst = [pd,pf,prec,g]
    nextpath = join(path, one)
    data = [join(nextpath, f)
            for f in listdir(nextpath) if isfile(join(nextpath, f))]
    predict = data.pop(-1)
    tune_predict = data.pop(-1)
    train = data
    global The
    The.data.predict = tune_predict
    The.data.train = train
    The.option.baseLine = False
    writefile("Begin time :" +strftime("%Y-%m-%d %H:%M:%S"))
    writefile("Dataset: "+one)
    cart_tunetest(predict)
    cart_basetest(predict)
    rndfsttest(predict)
    printresult(one)


if __name__ =="__main__":
	eval(cmd()) 