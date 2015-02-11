
from models import *
from path import path
path = str(path("./defectprediction/").abspath())
sys.path.insert(0, path)
from Abcd import *
def _Abcd(predicted, actual):
  predicted_txt = []
  abcd = Abcd(db='Traing', rx='Testing')

  def isDef(x):
    return "Defective" if x > 0 else "Non-Defective"
  for data in predicted:
    predicted_txt +=[isDef(data)]
  for act, pre in zip(actual, predicted_txt):
    abcd.tell(act, pre)
  abcd.header()
  score = abcd.ask()
  # pdb.set_trace()
  return score

def buildtablefromfile(l):
	def build(file):
		from path import path
		path = str(path(file).abspath())
		with open(path) as f:contents = f.readlines()
		independent = []
		dependent = []
		for content in contents[1:]:
			independent.append([float(x) for x in content.split(',')[3:-1]])
			dependent.append(float(content.split(',')[-1]))
		return independent,dependent
	if isinstance(l,list):
		ind,dep = [],[]
		for file in l:
			t1,t2 = build(file)
			ind += t1
			dep += t2
		return ind,dep
	else:
		return build(l)
def weitransform(list):
	result = []
	for l in list:
		if l >0: result.append("Defective")
		else: result.append("Non-Defective")
	return result

class TunedRF(ModelBasic):
	def __init__(self,train,predict,n=4,objf=1):
		self.minR=[2,2,100, 1]
		self.maxR=[32,32,3000, 20]
		self.n = n
		self.minVal = 1e6
		self.maxVal = -1e6
		self.objf = objf
		self.past = [Log() for count in xrange(objf)]
		self.present = [Log() for count in xrange(objf)]
		self.lives=myModeloptions['Lives']
		self.functionDict = {}
		self.functionDict["f1"]="f1"
		self.train = train
		self.predict = predict

	def f1(self,listpoints,num=0):
		assert(len(listpoints) != 0),"parameters are empty"
		return runRF(listpoints,self.train,self.predict)

	def baseline(self,minR,maxR):
	    emin = 1e6
	    emax = -1e6
	    for x in range(0,90000):
	      solution = [(self.minR[z] + random.random()*(self.maxR[z]-self.minR[z])) for z in range(0,self.n)]
	      result=0
	      for i in xrange(self.objf):
	        temp="f"+str(i+1)
	        callName = self.functionDict[temp]
	        result+=float(getattr(self, callName)(solution,i+1))
	      #self.returnMax(result)
	      #self.returnMin(result)
	      emin = emin if emin < result else result
	      emax = emax if emax > result else result
	    return emin,emax


def runRF(listpoint,train,predict):

	indep,dep = buildtablefromfile(train)
	from sklearn.ensemble import RandomForestClassifier
	rf = RandomForestClassifier(min_samples_split = int(listpoint[0]),min_samples_leaf = int(listpoint[1]), n_estimators = int(listpoint[2]), max_features = int(round(listpoint[3])))
	rf.fit(indep, dep)


	actual_indep,actual_dep = buildtablefromfile(predict)
	arr = rf.predict(actual_indep)
	#pdb.set_trace()
	result = [i for i in arr]
	# for x in xrange(len(result)):
	# 	if result[x] != actual_dep[x]:
	# 		print result[x],actual_dep[x],x
	scores = _Abcd(result,weitransform(actual_dep))
	#print ">>>>>>>>>>>>>>>>>>>>>>>>> ",scores[-1][-1] 
	#scores = _Abcd([1,1,1,1,1],["Defective","Defective","Defective","Defective","Defective"])
	return scores[-1][-1]


def NaiveRF(train,predict):
	indep,dep = buildtablefromfile(train)
	from sklearn.ensemble import RandomForestClassifier
	rf = RandomForestClassifier()
	rf.fit(indep, dep)


	actual_indep,actual_dep = buildtablefromfile(predict)
	arr = rf.predict(actual_indep)
	#pdb.set_trace()
	result = [i for i in arr]
	# for x in xrange(len(result)):
	# 	if result[x] != actual_dep[x]:
	# 		print result[x],actual_dep[x],x
	scores = _Abcd(result,weitransform(actual_dep))
	#print ">>>>>>>>>>>>>>>>>>>>>>>>> ",scores[-1][-1] 
	#scores = _Abcd([1,1,1,1,1],["Defective","Defective","Defective","Defective","Defective"])
	return scores[-1][-1]
