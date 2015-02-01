from __future__ import division 
import sys
sys.dont_write_bytecode = True

myoptions = {'MaxWalkSat':{'maxTries':'50','maxChanges':'2000','threshold':'0.001','probLocalSearch':'0.25'},'SA':{'kmax':'1000','emax':'0'},'GA':{'crossOverRate':'0.6','mutationRate':'0.1','elitism':'50','generation':'20'},'DE':{'repeat':100,'np':100,'f':0.75,'cf':0.3},'PSO':{'N':30,'W':1,'phi1':1.3,'phi2':2.7,'repeat':1000,'threshold':'0.001'},'Seive':{'tries':'20','repeat':'6','intermaxlimit':'20','extermaxlimit':'20','threshold':'15','initialpoints':'1000','lives':'4','subsample':'10'}}
myModeloptions = {'Lives': 4,'a12':0.56}
myModelobjf = {'Viennet':3,'Schaffer':2, 'Fonseca':2, 'Kursawe':2, 'ZDT1':2,'ZDT3':2,'DTLZ7':20,'Schwefel':1,'Osyczka':2,'DTLZ1':20,'DTLZ2':10,'DTLZ3':10,'DTLZ4':10,'DTLZ5':10,'DTLZ6':10}
