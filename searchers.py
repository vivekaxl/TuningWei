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

#say = Utilities().say


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

    frontier = [ self.generate_point(model) for _ in xrange(np)]

    for i in xrange(repeat):
      if lives == 0:  
        print
        break
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

