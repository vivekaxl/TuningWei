from __future__ import division, print_function
from main import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score

def csv2py(f):
  sym2num = {}
  # sym2num hold all the characters with assinged numbers that never seen

  def str2num(t, p=0):
    def bigt():
      if isinstance(tbl, list):
        t = tbl[0]
        for i in range(1, len(tbl)):
          t._rows += tbl[i]._rows
      else:
        t = tbl
      return t
    t = bigt()
    for r, row in enumerate(t._rows):
      for c, cell in enumerate(row.cells):
        if isinstance(cell, str) and c < t.depen[0].col and isinstance(t.headers[c], Sym):
          if sym2num.get(cell, 0) == 0:
            sym2num[cell] = p
            p += 1
          t._rows[r].cells[c] = sym2num[cell]  # update cell with num
    return t
  if isinstance(f, list):
    tbl = [table(src) for src in f]  # tbl is a list of tables
  else:
    tbl = table(f)
  tbl_num = str2num(tbl)
  return tbl_num

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

def conv(x):
  return [ float(i) for i in x]
def cart():
  clf = None
  if The.classifier.tuned:
    # classifier = DecisionTreeClassifier 
    # clf = classifier(criterion = The.cart.criterion, random_state = 0)
    clf = DecisionTreeClassifier(criterion = The.cart.criterion, 
        max_features = The.cart.max_features, max_depth = The.cart.max_depth, 
        min_samples_split = The.cart.min_samples_split, min_samples_leaf = The.cart.min_samples_leaf,
        max_leaf_nodes = The.cart.max_leaf_nodes, random_state= 0)
  elif The.classifier.cart:
    clf = DecisionTreeClassifier(random_state=0)
  elif The.classifier.rdfor:
    clf = RandomForestClassifier(n_estimators = 100)
	# The.data.train =["./data/ant/ant-1.4.csv"]
	# The.data.predict ="./data/ant/ant-1.5.csv"
  testdata, actual = buildtestdata1(The.data.predict)
  traintable= csv2py(The.data.train)
  traindata_X = [ conv(row.cells[:-1])for row in traintable._rows]
  traindata_Y = [ (row.cells[-1])for row in traintable._rows]
  # pdb.set_trace()
  predictdata_X =[ conv(row.cells[:-1])for row in testdata]
  predictdata_Y =[ (row.cells[-1]) for row in testdata]
  clf = clf.fit(traindata_X, traindata_Y)
  array = clf.predict(predictdata_X)
  predictresult = [i for i in array]
  # print(predictresult)
  # print(predictdata_Y)
  scores = _Abcd(predictresult,actual)
  return scores



# def main():
# 	return cart()

if __name__ == "__main__":
	eval(cmd())
