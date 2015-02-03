
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

def NaiveRF(train,predict):
	indep,dep = buildtablefromfile(train)
	from sklearn.ensemble import RandomForestClassifier
	rf = RandomForestClassifier(n_estimators=100)
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
