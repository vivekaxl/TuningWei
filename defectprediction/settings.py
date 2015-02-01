import sys
sys.dont_write_bytecode = True 
from demos import *
from settingswhere2 import *

class Thing(object):
  id = -1
  def __init__(i,**fields) : 
    i.override(fields)
    i.newId()
  def newId(i):
    i._id = Thing.id = Thing.id + 1
  def also(i,**d)  : return i.override(d)
  def override(i,d): i.__dict__.update(d); return i
  def update(i,**d) : i.__dict__.update(d); return i
  def __hash__(i) : return i._id
 # def __eq__(i,j)  : return i._id == j._id
  #def __neq__(i,j) : return i._id != j._id

The = Thing()
def settings(f=None):
  if f : The.__dict__[f.func_name[:-4]] = f() 
  else : rprintln(The)
  return f

@settings
def stringings(): return Thing(
  white= r'["\' \t\r\n]')

@settings
def mathings(): return Thing(
  seed  = 1,
  inf   =    10**32,
  ninf  = -1*10**32,
  teeny =    10**-32,
  bootstraps = 500,
  a12   = Thing(
    small   = [.6, .68][0],
    reverse = False),
  brink = Thing(
    hedges= [ .39, 1.0 ][0], 
    cohen = [ .3 ,  .5 ][0],
    conf  = [ .95,  .99][0]))

@settings
def sampleings(**d): return Thing(
  keep = 256,
  bins = 5,
  tiny = 0.1,
  enough=4).override(d)

@settings
def readerings(): return Thing(
  sep      = ",",
  bad      = r'(["\' \t\r\n]|#.*)',
  skip     ='?',
  showonly = '-',
  numc     ='$',
  missing  = '?',
  patterns = {
    '\$'     : lambda z: z.nums,
    '\.'     : lambda z: z.syms,
    '>'      : lambda z: z.more,
    '<'      : lambda z: z.less,
    '='      : lambda z: z.klass,
    '[=<>]'  : lambda z: z.depen,
    '^[^=<>].*[^/]$': lambda z: z.indep,
    '.'      : lambda z: z.headers})

@settings
def treeings(**d): return Thing(
  min=4,
  infoPrune=0.33,
  variancePrune=True,
  debug=False,
  m=5,
  n=5,
  missing = '?',
  better  = lambda x: x.better,
  worse  = lambda x: x.worse,
  cells = lambda x: x.cells,
  prune=False
  ).override(d)

@settings
def distings(**d): return Thing(
  cells   = lambda x : x.cells,
  what    = lambda x : x.indep,
  missing = '?',
  deep    = 10,
  repeats = 1,
  verbose = False,
  cache   = True,
  some    = None,
  err     = lambda p,a: abs(p-a)/(a+0.001),
  tiny    = lambda t: len(t._rows)**0.5,
  klass   = lambda x,t,o:x.cells[t.klass[0].col],
  retry   = 10
  ).override(d)

@settings
def optionings(**d):return Thing(
  # showWhere = False, 
  showDTree = False, 
  showTogo = False,
  showConstrast = False,
  showSummarize=False,
  threshold = 0.1,
  tuning = True,
  clustering = False,
  # treeMin = 0.5,
  minSize = 0.5,
  baseLine = False, 
  k = 1,
  mean = True,
  count = 0
  ).override(d)

@settings
def whereings(**d):return Thing(
  minSize  = 10,    # min leaf size
  depthMin= 2,      # no pruning till this depth
  depthMax= 10,     # max tree depth
  wriggle = 0.2,    # min difference of 'better'
  prune   = True,   # pruning enabled?
  b4      = '|.. ', # indent string
  verbose = False,  # show trace info?
  goal    = lambda m,x : scores(m,x),
  seed    = 1,
  cache   = Thing(size=128)
  ).override(d)

@settings
def cartings(**d): return Thing(
  criterion = "entropy",
  splitter = "best",
  max_features = None,
  max_depth = None,
  min_samples_split = 2,
  min_samples_leaf = 1,
  max_leaf_nodes = None,
  random_state = 0
  ).override(d)


@settings
def dataings(**d):return Thing(
  # predict = "",
  predict = "",
  train = "",
  ).override(d)
@settings
def classifierings(**d):return Thing(
  cart = False,
  tuned = False,
  rdfor = False
  ).override(d)

@demo
def thingsed(**d):
  import lib
  lib.rprintln(The)

if __name__ == '__main__': eval(cmd())
