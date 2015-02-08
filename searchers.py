from __future__ import division 
import sys 
import random
import math 
import numpy as np
from where_mod import *
#from where_tm import *
from models import *
from options import *
from utilities import *
sys.dont_write_bytecode = True

def say(x): 
  sys.stdout.write(str(x))
  sys.stdout.flush()


class SearchersBasic():
  tempList=[]
  def display(self,score,printChar=''):
    self.tempList.append(score)
    if(self.displayStyle=="display1"):
      print(printChar),
  
  def display2(self):
    if(self.displayStyle=="display2"):
      #print xtile(self.tempList,width=25,show=" %1.6f")
      self.tempList=[]

class Points():
  dec = []
  obj = []
  score = -1e6
  def __init__(i,dec,obj) :
    i.dec = dec
    i.obj = obj
    i.score = np.sum(obj)
  def __eq__(i,j)  : 
    from collections import Counter
    return Counter(i.dec) == Counter(j.dec)
  def __ne__(i,j)  :
    from collections import Counter
    return Counter(i.dec) == Counter(j.dec)  
  def __repr__(i)  : return '{' + showd(i.__dict__) + '}'

class DE(SearchersBasic):
  def __init__(self,modelName,displayS,bmin,bmax):
    self.model=modelName
    self.displayStyle=displayS
    self.model.minVal = bmin
    self.model.maxVal = bmax

  def generate_point(self,model,decns=[]):
    if len(decns) == 0:
      decns =[model.minR[i]+random.random()*(model.maxR[i]-model.minR[i]) for i in xrange(model.n)]
      return Points(
             dec = list(decns),
             obj = model.evaluate(decns)[:-1]
             )
    else:
      return Points(
             dec = list(decns),
             obj = model.evaluate(decns)[:-1]
             )

  def threeOthers(self,frontier,one):
    #print "threeOthers"
    seen = [one]
    def other():
      #print "other"
      for i in xrange(len(frontier)):
        while True:
          k = random.randint(0,len(frontier)-1)
          #print "%d"%k
          if frontier[k] not in seen:
            seen.append(frontier[k])
            break
        return frontier[k]
    this = other()
    that = other()
    then = other()
    return this,that,then
  
  def trim(self,x,i)  : # trim to legal range
    m=self.model
    return max(m.minR[i], min(x, m.maxR[i]))      

  def extrapolate(self,frontier,one,f,cf):
    #print "Extrapolate"
    two,three,four = self.threeOthers(frontier,one)
    #print two,three,four
    solution=[]
    for d in xrange(self.model.n):
      x,y,z=two.dec[d],three.dec[d],four.dec[d]
      if(random.random() < cf):
        solution.append(self.trim(x + f*(y-z),d))
      else:
        solution.append(one.dec[d])
    return self.generate_point(model = self.model,decns = solution)

  def update(self,m,f,cf,frontier,minScore=1e6,total=0.0,n=0):
    def lo(m,index)      : return m.minR[index]
    def hi(m,index)      : return m.maxR[index]

    def trim(m,x,i)  : # trim to legal range
      temp = min(hi(m,i),max(lo(m,i),x))
      assert( lo(m,i) <= temp and hi(m,i) >= temp),"error"
      return temp

    def better(old,new):
      assert(len(old)==len(new)),"Length mismatch"
      for o,n in zip(old,new): #Since the score is return as [values of all objectives and energy at the end]
        if o <= n: pass
        else: return False
      return True

    changed = False
    model=self.model
    newF = []
    total,n=0,0
    mean_score = []
    for x in frontier:
      new = self.extrapolate(frontier,x,f,cf)
      if better(x.obj,new.obj) == True and x.score < new.score:
        newF.append(new)
        mean_score.append(x.score)
        changed = True
      else:
        newF.append(x)
        mean_score.append(x.score)

    return newF,changed
      
  def evaluate(self,repeat=100,np=50,f=0.8,cf=0.9,epsilon=0.01,lives=5):
    #print "evaluate"
    model=self.model
    minR = model.minR
    maxR = model.maxR
    np = model.n * 10 # from storn's recommendation
    print "Options: \n Generation: ",repeat
    print "Frontier size: ",np
    print "f: ",f
    print "cf: ",cf
    print "Lives: ",lives
    frontier = [ self.generate_point(model) for _ in xrange(np)]

    for xyz in xrange(repeat):
      if lives == 0:  
        print
        break
      say(xyz)
      frontier,changed = self.update(model,f,cf,frontier)

      self.model.evalBetter()
      if changed == False: 
        lives -= 1
        print "lost it",

    #minR=9e10
    maxno = -1e6
    for x in frontier:
      #print x
      energy = x.score
      if maxno < energy:
        maxno = energy
        solution = x
      # if(minR>energy):
      #   minR = energy
      #   solution=x 
    return solution,self.model


class Seive2_Initial(SearchersBasic):
  def __init__(self,modelName,displayS,bmin,bmax):
    self.model = modelName
    self.model.minVal = bmin
    self.model.maxVal = bmax
    self.displayStyle=displayS
    self.threshold = int(myoptions['Seive2_Initial']['threshold'])         
    self.ncol=8               #number of columns in the chess board
    self.nrow=8               #number of rows in the chess board
    self.intermaxlimit=int(myoptions['Seive2_Initial']['intermaxlimit'])     #Max number of points that can be created by interpolation
    self.extermaxlimit=int(myoptions['Seive2_Initial']['extermaxlimit'])     #Max number of points that can be created by extrapolation
    self.evalscores=0

  def fastmap(self,model,data):
    "Divide data into two using distance to two distant items."
    #print ">>>>>>>>>>>>>>>>>>.FastMap"
    #print "Length of data: ",len(data)
    one  = any(data)             # 1) pick anything
    west = furthest(model,one,data)  # 2) west is as far as you can go from anything
    east = furthest(model,west,data) # 3) east is as far as you can go from west
    c    = dist(model,west,east)
    # now find everyone's distance
    xsum, lst = 0.0,[]
    ws = score(model,west)[-1]
    es = score(model,east)[-1]
    #print "West: ",ws
    #print "East: ",es
    for one in data:
      a = dist(model,one,west)
      b = dist(model,one,east)
      x = (a*a + c*c - b*b)/(2*c) # cosine rule
      xsum += x
      lst  += [(x,one)]
    # now cut data according to the mean distance
    if ws > es:
      cut, wests, easts = xsum/len(data), [], []
      for x,one in lst:
        where = wests if x < cut else easts 
        where += [one]
      #its assumed east is heaven
      return [self.gale_mutate(model,point,c,east,west) for point in easts]
    else:
      cut, wests, easts = xsum/len(data), [], []
      for x,one in lst:
        where = wests if x < cut else easts 
        where += [one]
        #its assumed east is heaven
      return [self.gale_mutate(model,point,c,west,east) for point in easts]


  def gale_mutate(self,model,point,c,east,west,multiplier = 4.5):
    #tooFar = multiplier * abs(c)
    #print "C: ",c
    tooFar = multiplier * abs(c)
    import copy
    new = copy.deepcopy(point)
    for i in xrange(len(point.dec)):
      d = east.dec[i] - west.dec[i]
      if not d == 0:
        d = -1 if d < 0 else 1
        #d = east.dec[i] = west.dec[i]
        x = new.dec[i] * (1 + abs(c) * d)
        new.dec[i] = max(min(hi(model,i),x),lo(model,i))
        # if x != new.dec[i] : print "blah",new.dec[i]-x
        # else: print "boom"
    newDistance = self.project(model,west,east,c,new) -\
                  self.project(model,west,east,c,west)
    
    if abs(newDistance) < tooFar  and self.valid(model,new):
      return new
    else:
      # print "Distance: ",abs(newDistance), "toofar: ",abs(tooFar)
      #print "Blown away"
      return point

  def tgenerate(self,m,pop,gen=0):
    it = int(myoptions['Seive2_Initial']['tgen'])
    for _ in xrange(it):
      temp = random.random()
      o = any(pop)
      t = any(pop)
      th = any(pop)
      if temp <= 0.5:  cand = polate(m,o.dec,t.dec,th.dec,0.1,0.5)
      else: cand = polate(m,o.dec,t.dec,th.dec,0.9,2.0)
      one = self.generateSlot(m,cand,-1,-1)
      #print one.dec
      pop += [one]
    return pop

  def polate(m,lx,ly,lz,fmin,fmax):
    def lo(m,index)      : return m.minR[index]
    def hi(m,index)      : return m.maxR[index]
    def trim(m,x,i)  : # trim to legal range
      temp = min(hi(m,i),max(lo(m,i),x))
      assert( lo(m,i) <= temp and hi(m,i) >= temp),"error"
      return temp
    def indexConvert(index):
      return int(index/100),index%10

    assert(len(lx)==len(ly)==len(lz))
    cr=0.3
    genPoint=[]
    for i in xrange(len(lx)):
      x,y,z = lx[i],ly[i],lz[i]
      rand = random.random()

      if rand < cr:
        probEx = fmin + (fmax-fmin)*random.random()
        new = trim(m,x + probEx*(y-z),i)
      else:
        new = y #Just assign a value for that decision
      genPoint.append(new)
    return genPoint
  def project(self,model,west, east, c, x):
    "Project x onto line east to west"
    if c == 0: return 0
    a = dist(model,x,west)
    b = dist(model,x,east)
    return (a*a + c*c - b*b)/(2*c) # cosine rule

  def valid(self,m,val):
    for x in xrange(len(val.dec)):
      if not m.minR[x] <= val.dec[x] <= m.maxR[x]: 
        return False
    return True

  def generate2(self,model,constraints):
    def any(l,h):
      #print ">>>>>>>> : ",lo,hi  
      return (l + random.random()*(h-l))
    points = []
    for _ in xrange(30):
      for _ in xrange(400):
        dec = []
        for constraint in constraints:
          #lo,hi = self.one(model,constraint )
          #print constraint
          lo,hi = constraint[1][0],constraint[1][1]
          temp = any(lo,hi)
          assert(temp >= lo and temp <= hi),"ranges are messed up"
          dec.append(temp)
        points.append(self.generateSlot(model,dec,-1,-1))
      #print "After Generation: ",len(points)
      points = self.fastmap(model,points)
      points += return_points(model,100)
      #print "After FastMap: ",len(points)
      points = self.tgenerate(model,points)
    #print ">>>>>>>Final: ",len(points)
    #raise Exception("asdasdasffd")
    #print "\n\n",points
    #assert(len(points) == 940),"all the points were not generated"
    return points


  def evaluate(self,points=[],depth=4):
    def generate_dictionary(points=[]):  
      dictionary = {}
      chess_board = whereMain(self.model,points) #checked: working well
      #print chess_board
      for i in range(1,9):
        for j in range(1,9):
          temp = [x for x in chess_board if x.xblock==i and x.yblock==j]
          if(len(temp)!=0):
            index=temp[0].xblock*100+temp[0].yblock
            dictionary[index] = temp
            assert(len(temp)==len(dictionary[index])),"something"
      #print dictionary.keys()
      return dictionary

    def thresholdCheck(index,dictionary):
      try:
        #print "Threshold Check: ",index
        if(len(dictionary[index])>self.threshold):return True
        else:return False
      except:
        return False
    def indexof(lsts,number,index = lambda x: x[1]):
      for i,lst in enumerate(sorted(lsts,key = index)):   
        if number == index(lst): return i
      return -1 
    def uscore(lsts,starti,endi):
      summ = 0
      for i in xrange(starti,endi+1):
        summ += lsts[i][-1]
      return summ/(endi-starti+1)

    model = self.model
    minR = model.minR
    maxR = model.maxR
    #if depth == 0: model.baseline(minR,maxR)

          #if depth == 0 and len(points) == 0: 
        #generate points according to the constraints
    print "return points is here"
    points = return_points(model,60)
    for point in points:
      point.score = scores(model,point)[-1]
    
    points = [point.dec+[point.score] for point in points]
    constraints = []
    for i in xrange(len(points[0])-1):
      constraint = []
      cohen=0.3
      h = 1e6
      #print self.sdiv(points,cohen=cohen,num1=lambda x:x[i],num2=lambda x:x[-1])[0][1][i]
      #print "........................>>",len(self.sdiv(points,cohen=cohen,num1=lambda x:x[i],num2=lambda x:x[-1]))
      for d in  self.sdiv(points,cohen=cohen,num1=lambda x:x[i],num2=lambda x:x[-1]):
        starti = indexof([point[:-1] for point in points],d[1][0][i],lambda x:x[i])
        endi =  indexof([point[:-1] for point in points],d[1][-1][i],lambda x:x[i])
        #print "Starti: ",starti, "Endi: ",endi
        mean_score = uscore(sorted(points,key = lambda x:x[i]),starti,endi)
        #print "-----------------Mean Score: ",mean_score
        if mean_score < h:
          const1 = d[1][0][i]
          const2 = d[1][-1][i]
          h = mean_score
          #print "+++++++++++++++High Mean Score: ",h
      #constraint.append()

      
      constraints.append([i]+[(const1,const2)])  
    # print constraints
    # raise Exception(":asd")    
    points = self.generate2(model,constraints)
    #print "Number of points: ",len(points)
    print model.no_eval
    dictionary = generate_dictionary(points)
    # for key in dictionary.keys():
    #   try:
    #     print "Key: ",key, "Number: ",len(dictionary[key])
    #   except:
    #     print "Empty"


    from collections import defaultdict
    graph = defaultdict(list)
    matrix = [[0 for x in range(8)] for x in range(8)]
    for i in xrange(1,9):
      for j in xrange(1,9): 
        if(thresholdCheck(i*100+j,dictionary)==False):
          result,dictionary = self.generateNew(self.model,i,j,dictionary)
          if result == False: 
            matrix[i-1][j-1] = 100
            print "in middle of desert"
            continue
        matrix[i-1][j-1] = score(model,self.one(model,dictionary[i*100+j]))[-1]

        
       # print matrix[i-1][j-1],
      #print
    for i in xrange(1,9):
      for j in xrange(1,9):
        sumn=0
        s = matrix[i-1][j-1]
        neigh = self.listofneighbours(i,j)
        sumn = sum([1 for x in neigh if matrix[self.rowno(x)-1][self.colmno(x)-1]>s])
        if (i*100+j) in dictionary:
          graph[int(sumn)].append(i*100+j)
        
    high = -1e6
    bsoln = None
    if len(graph.keys()) != 0:
      maxi = max(graph.keys())
      #print graph.keys()
      #print "Number of points: ",len(graph[maxi])
      for x in graph[maxi]:
         #print "Seive2:B Number of points in ",maxi," is: ",len(dictionary[x])
         #if(len(dictionary[x]) < 15: [self.n_i(model,dictionary,x) for _ in xrange(20)]
         #print "Seive2:A Number of points in ",maxi," is: ",len(dictionary[x])
         for y in dictionary[x]:
           tempscores = scores(model,y)
           print tempscores[-1]
           if tempscores[-1] > high:
             high = tempscores[-1]
             bsoln = y
      #print count  
    print bsoln,tempscores[-1]   
    return Points(dec = bsoln.dec, obj = tempscores[:-1]),model

  def one(self,model,lst): 
    def any(l,h):
      return (0 + random.random()*(h-l))
    return lst[int(any(0,len(lst) - 1)) ]

  def convert(self,x,y): return (x*100)+y
  def rowno(self,x): return int(x/100)
  def colmno(self,x): return x%10 

  def gonw(self,x):
    nrow=self.nrow
    ncol=self.ncol
    if(self.rowno(x)==1 and self.colmno(x)==1):return self.convert(nrow,ncol)#in the first coulumn and first row
    elif(self.rowno(x)==1): return self.convert(nrow,self.colmno(x)-1)
    elif(self.colmno(x)==1): return self.convert(self.rowno(x)-1,ncol)#in the first column
    else: return (x-101)

  def gow(self,x):
    nrow=self.nrow
    ncol=self.ncol
    if(self.colmno(x)==1): return self.convert(self.rowno(x),ncol)
    else: return (x-1)

  def gosw(self,x):
    nrow=self.nrow
    ncol=self.ncol
    if(self.rowno(x)==nrow and self.colmno(x)==1): return self.convert(1,ncol)
    elif(self.rowno(x)==nrow): return self.convert(1,self.colmno(x)-1)
    elif(self.colmno(x)==1): return self.convert(self.rowno(x)+1,ncol)
    else: return (x+99)

  def gos(self,x):
    nrow=self.nrow
    ncol=self.ncol
    if(self.rowno(x)==nrow): return self.convert(1,self.colmno(x))
    else: return x+100

  def gose(self,x):
    nrow=self.nrow
    ncol=self.ncol
    if(self.rowno(x)==nrow and self.colmno(x)==ncol): return self.convert(1,1)
    elif(self.rowno(x)==nrow): return self.convert(1,self.colmno(x)+1)
    elif(self.colmno(x)==ncol): return self.convert(self.rowno(x)+1,1)
    else: return x+101

  def goe(self,x):
    nrow=self.nrow
    ncol=self.ncol
    if(self.colmno(x)==ncol): return self.convert(self.rowno(x),1)
    else: return x+1

  def gone(self,x):
    nrow=self.nrow
    ncol=self.ncol
    if(self.rowno(x)==1 and self.colmno(x)==ncol): return self.convert(nrow,1)
    elif(self.rowno(x)==1): return self.convert(nrow,self.colmno(x)+1)
    elif(self.colmno(x)==ncol): return self.convert(self.rowno(x)-1,1)
    else: return x-99

  def gon(self,x):
    nrow=self.nrow
    ncol=self.ncol
    if(self.rowno(x)==1): return self.convert(nrow,self.colmno(x))
    else: return x-100 

  def getpoints(self,index,dictionary):
    tempL = []
    for x in dictionary[index]:tempL.append(x.dec)
    return tempL

  def wrapperInterpolate(self,m,xindex,yindex,maxlimit,dictionary):
    def interpolate(lx,ly,cr=0.3,fmin=0,fmax=1):
      def lo(m,index)      : return m.minR[index]
      def hi(m,index)      : return m.maxR[index]
      def trim(m,x,i)  : # trim to legal range
        return max(lo(m,i), x%hi(m,i))
      assert(len(lx)==len(ly))
      genPoint=[]
      for i in xrange(len(lx)):
        x,y=lx[i],ly[i]
        #print x
        #print y
        rand = random.random
        if rand < cr:
          probEx = fmin +(fmax-fmin)*rand()
          new = trim(m,min(x,y)+probEx*abs(x-y),i)
        else:
          new = y
        genPoint.append(new)
      return genPoint

    decision=[]
    #print "Number of points in ",xindex," is: ",len(dictionary[xindex])
    #print "Number of points in ",yindex," is: ",len(dictionary[yindex])
    xpoints=self.getpoints(xindex,dictionary)
    ypoints=self.getpoints(yindex,dictionary)
    import itertools
    listpoints=list(itertools.product(xpoints,ypoints))
    count=0
    while True:
      if(count>min(len(xpoints),maxlimit)):break
      x=self.one(m,listpoints)
      temp = interpolate(x[0],x[1])
      decision.append(temp)
      count+=1
    
    return decision

  def wrapperextrapolate(self,m,xindex,yindex,maxlimit,dictionary):
    def extrapolate(lx,ly,lz,cr=0.3,fmin=0.9,fmax=2):
      def lo(m,index)      : return m.minR[index]
      def hi(m,index)      : return m.maxR[index]
      def trim(m,x,i)  : # trim to legal range
        return max(lo(m,i), x%hi(m,i))
      def indexConvert(index):
        return int(index/100),index%10
      assert(len(lx)==len(ly)==len(lz))
      genPoint=[]
      for i in xrange(len(lx)):
        x,y,z = lx[i],ly[i],lz[i]
        rand = random.random()

        if rand < cr:
          probEx = fmin + (fmax-fmin)*random.random()
          new = trim(m,x + probEx*(y-z),i)
        else:
          new = y #Just assign a value for that decision
        genPoint.append(new)
      return genPoint

    decision=[]
    #TODO: need to put an assert saying checking whether extrapolation is actually possible
    xpoints=self.getpoints(xindex,dictionary)
    ypoints=self.getpoints(yindex,dictionary)
    count=0
    while True:
      if(count>min(len(xpoints),maxlimit)):break
      two = self.one(m,xpoints)
      index2,index3=0,0
      while(index2 == index3): #just making sure that the indexes are not the same
        index2=random.randint(0,len(ypoints)-1)
        index3=random.randint(0,len(ypoints)-1)

      three=ypoints[index2]
      four=ypoints[index3]
      temp = extrapolate(two,three,four)
      #decision.append(extrapolate(two,three,four))
      decision.append(temp)
      count+=1
    return decision

  def generateNew(self,m,xblock,yblock,dictionary,flag = False):
    convert = self.convert
    rowno = self.rowno
    colmno = self.colmno 

    def indexConvert(index):
      return int(index/100),index%10

    def opposite(a,b):
      ax,ay,bx,by=a/100,a%100,b/100,b%100
      if(abs(ax-bx)==2 or abs(ay-by)==2):return True
      else: return False

    def thresholdCheck(index):
      try:
        #print "Threshold Check: ",index
        if(len(dictionary[index])>self.threshold):return True
        else:return False
      except:
        return False

    def interpolateCheck(xblock,yblock):
      returnList=[]
      if(thresholdCheck(self.gonw(convert(xblock,yblock))) and thresholdCheck(self.gose(convert(xblock,yblock))) == True):
        returnList.append(self.gonw(convert(xblock,yblock)))
        returnList.append(self.gose(convert(xblock,yblock)))
      if(thresholdCheck(self.gow(convert(xblock,yblock))) and thresholdCheck(self.goe(convert(xblock,yblock))) == True):
       returnList.append(self.gow(convert(xblock,yblock)))
       returnList.append(self.goe(convert(xblock,yblock)))
      if(thresholdCheck(self.gosw(convert(xblock,yblock))) and thresholdCheck(self.gone(convert(xblock,yblock))) == True):
       returnList.append(self.gosw(convert(xblock,yblock)))
       returnList.append(self.gone(convert(xblock,yblock)))
      if(thresholdCheck(self.gon(convert(xblock,yblock))) and thresholdCheck(self.gos(convert(xblock,yblock))) == True):
       returnList.append(self.gon(convert(xblock,yblock)))
       returnList.append(self.gos(convert(xblock,yblock)))
      return returnList


    def extrapolateCheck(xblock,yblock):
      #TODO: If there are more than one consequetive blocks with threshold number of points how do we handle it?
      #TODO: Need to make this logic more succint
      returnList=[]
      #go North West
      temp = self.gonw(convert(xblock,yblock))
      result1 = thresholdCheck(temp)
      if result1 == True:
        result2 = thresholdCheck(self.gonw(temp))
        if(result1 == True and result2 == True):
          returnList.append(temp)
          returnList.append(self.gonw(temp))

      #go North 
      temp = self.gon(convert(xblock,yblock))
      result1 = thresholdCheck(temp)
      if result1 == True:
        result2 = thresholdCheck(self.gon(temp))
        if(result1 == True and result2 == True):
          returnList.append(temp)
          returnList.append(self.gon(temp))

      #go North East
      temp = self.gone(convert(xblock,yblock))
      result1 = thresholdCheck(temp)
      if result1 == True:
        result2 = thresholdCheck(self.gone(temp))
        if(result1 == True and result2 == True):
          returnList.append(temp)
          returnList.append(self.gone(temp))
  
      #go East
      temp = self.goe(convert(xblock,yblock))
      result1 = thresholdCheck(temp)
      if result1 == True:
        result2 = thresholdCheck(self.goe(temp))
        if(result1 == True and result2 == True):
          returnList.append(temp)
          returnList.append(self.goe(temp))

      #go South East
      temp = self.gose(convert(xblock,yblock))
      result1 = thresholdCheck(temp)
      if result1 == True:
        result2 = thresholdCheck(self.gose(temp))
        if(result1 == True and result2 == True):
          returnList.append(temp)
          returnList.append(self.gose(temp))

      #go South
      temp = self.gos(convert(xblock,yblock))
      result1 = thresholdCheck(temp)
      if result1 == True:
        result2 = thresholdCheck(self.gos(temp))
        if(result1 == True and result2 == True):
          returnList.append(temp)
          returnList.append(self.gos(temp))

      #go South West
      temp = self.gosw(convert(xblock,yblock))
      result1 = thresholdCheck(temp)
      if result1 == True:
        result2 = thresholdCheck(self.gosw(temp))
        if(result1 == True and result2 == True):
          returnList.append(temp)
          returnList.append(self.gosw(temp))
 
      #go West
      temp = self.gow(convert(xblock,yblock))
      result1 = thresholdCheck(temp)
      if result1 == True:
        result2 = thresholdCheck(self.gow(temp))
        if(result1 == True and result2 == True):
          returnList.append(temp)
          returnList.append(self.gow(temp))

      return returnList
  
    newpoints=[]
    if flag == True:
      if convert(xblock,yblock) in dictionary: pass
      else:
        assert(convert(xblock,yblock)>=101),"Something's wrong!" 
        assert(convert(xblock,yblock)<=808),"Something's wrong!"
      decisions=[]
      listInter=interpolateCheck(xblock,yblock)
      #print "generateNew|Interpolation Check: ",listInter
      if(len(listInter)!=0):
        assert(len(listInter)%2==0),"listInter%2 not 0"
        for i in xrange(int(len(listInter)/2)):
          #print "FLAG is True!"
          decisions.extend(self.wrapperInterpolate(m,listInter[i*2],\
          listInter[(i*2)+1],1000,dictionary))
      #else:
        #print "generateNew| Interpolation failed"
      listExter = extrapolateCheck(xblock,yblock)
      #print "generateNew|Extrapolation Check: ",listInter
      if(len(listExter)!= 0):
        #print "generateNew| Extrapolation failed"
      #else:
        #print "FLAG is True!"
        for i in xrange(int(len(listExter)/2)):
            decisions.extend(self.wrapperextrapolate(m,listExter[2*i],\
            listExter[(2*i)+1],1000,dictionary))
      
      for decision in decisions:dictionary[convert(xblock,yblock)].\
      append(self.generateSlot(m,decision,xblock,yblock))
      return True,dictionary   


    #print "generateNew| convert: ",convert(xblock,yblock)
    #print "generateNew| thresholdCheck(convert(xblock,yblock): ",thresholdCheck(convert(xblock,yblock))
    #print "generateNew| points in the block: ",len(dictionary[convert(xblock,yblock)])
    if(thresholdCheck(convert(xblock,yblock))==False):
      #print "generateNew| Cell is relatively sparse: Might need to generate new points"
      listInter=interpolateCheck(xblock,yblock)
      #print "generateNew|Interpolation Check: ",listInter
      if(len(listInter)!=0):
        decisions=[]
        assert(len(listInter)%2==0),"listInter%2 not 0"
      #print thresholdCheck(xb),thresholdCheck(yb)
        for i in xrange(int(len(listInter)/2)):
            decisions.extend(self.wrapperInterpolate(m,listInter[i*2],listInter[(i*2)+1],int(self.intermaxlimit/len(listInter))+1,dictionary))

        if convert(xblock,yblock) in dictionary: pass
        else:
          #print convert(xblock,yblock)
          assert(convert(xblock,yblock)>=101),"Something's wrong!" 
          #assert(convert(xblock,yblock)<=808),"Something's wrong!" 
          assert(convert(xblock,yblock)<=808),"Something's wrong!"
          dictionary[convert(xblock,yblock)]=[]
        for decision in decisions:dictionary[convert(xblock,yblock)].append(self.generateSlot(m,decision,xblock,yblock))
        return True,dictionary
      else:
        #print "generateNew| Interpolation failed!"
        decisions=[]
        listExter = extrapolateCheck(xblock,yblock)
        #print "generateNew|Extrapolation Check: ",listInter
        if(len(listExter)==0):
          #print "generateNew|Interpolation and Extrapolation failed|In a tight spot..somewhere in the desert RANDOM JUMP REQUIRED"
          return False,dictionary
        else:
          assert(len(listExter)%2==0),"listExter%2 not 0"
          for i in xrange(int(len(listExter)/2)):
              decisions.extend(self.wrapperextrapolate(m,listExter[2*i],listExter[(2*i)+1],int(self.extermaxlimit)/len(listExter),dictionary))
          if convert(xblock,yblock) in dictionary: pass
          else: 
            assert(convert(xblock,yblock)>=101),"Something's wrong!" 
            #assert(convert(xblock,yblock)<=808),"Something's wrong!" 
            assert(convert(xblock,yblock)<=808),"Something's wrong!" 
            dictionary[convert(xblock,yblock)]=[]
          for decision in decisions: dictionary[convert(xblock,yblock)].append(self.generateSlot(m,decision,xblock,yblock))
          return True,dictionary
    else:
      listExter = extrapolateCheck(xblock,yblock)
      if(len(listExter) == 0):
        print "generateNew| Lot of points but middle of a desert"
        return False,dictionary #A lot of points but right in the middle of a deseart
      else:
        return True,dictionary


  """
  Return a list of neighbours:
  """
  def listofneighbours(self,xblock,yblock):
    index=self.convert(xblock,yblock)
    #print "listofneighbours| Index passed: ",index
    listL=[]
    listL.append(self.goe(index))
    listL.append(self.gose(index))
    listL.append(self.gos(index))
    listL.append(self.gosw(index))
    listL.append(self.gow(index))
    listL.append(self.gonw(index))
    listL.append(self.gon(index))
    listL.append(self.gone(index))
    return listL



  def generateSlot(self,m,decision,x,y):
    newpoint=Slots(changed = True,
            scores=1e6, 
            xblock=-1, #sam
            yblock=-1,  #sam
            x=-1,
            y=-1,
            obj = [None] * m.objf, #This needs to be removed. Not using it as of 11/10
            dec = [some(m,d) for d in xrange(m.n)])

    #scores(m,newpoint)
    #print "Decision: ",newpoint.dec
    #print "Objectives: ",newpoint.obj
    return newpoint

  def sdiv(self,lst, tiny=3,cohen=0.3,
             num1=lambda x:x[0], num2=lambda x:x[1]):
      "Divide lst of (num1,num2) using variance of num2."
      #----------------------------------------------
      class Counts(): # Add/delete counts of numbers.
        def __init__(i,inits=[]):
          i.zero()
          for number in inits: i + number 
        def zero(i): i.n = i.mu = i.m2 = 0.0
        def sd(i)  : 
          if i.n < 2: return i.mu
          else:       
            return (max(0,i.m2)*1.0/(i.n - 1))**0.5
        def __add__(i,x):
          i.n  += 1
          delta = x - i.mu
          i.mu += delta/(1.0*i.n)
          i.m2 += delta*(x - i.mu)
        def __sub__(i,x):
          if i.n < 2: return i.zero()
          i.n  -= 1
          delta = x - i.mu
          i.mu -= delta/(1.0*i.n)
          i.m2 -= delta*(x - i.mu)    
      #----------------------------------------------
      def divide(this,small): #Find best divide of 'this'
        lhs,rhs = Counts(), Counts(num2(x) for x in this)
        n0, least, cut = 1.0*rhs.n, rhs.sd(), None
        for j,x  in enumerate(this): 
          if lhs.n > tiny and rhs.n > tiny: 
            maybe= lhs.n/n0*lhs.sd()+ rhs.n/n0*rhs.sd()
            if maybe < least :  
              if abs(lhs.mu - rhs.mu) >= small: # where's the paper for this method?
                cut,least = j,maybe
          rhs - num2(x)
          lhs + num2(x)    
        return cut,least
      #----------------------------------------------
      def recurse(this, small,cuts):
        #print this,small
        cut,sd = divide(this,small)
        if cut: 
          recurse(this[:cut], small, cuts)
          recurse(this[cut:], small, cuts)
        else:   
          cuts += [(sd,this)]
        return cuts
      #---| main |-----------------------------------
      # for x in lst:
      #   print num2(x)
      small = Counts(num2(x) for x in lst).sd()*cohen # why we use a cohen??? how to choose cohen
      if lst: 
        return recurse(sorted(lst,key=num1),small,[])

