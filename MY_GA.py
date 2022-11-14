from tkinter import N
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.util.misc import repair
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback

#from pymoo.termination import get_termination


import numpy as np
import sys
import pickle
from pymoo.core.problem import ElementwiseProblem

from pymoo.util.display import Display

from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
#from pymoo.termination.default import DefaultMultiObjectiveTermination


import time

import sys
#sys.stdout = open('somefile.txt', 'w')
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

from pymoo.util.running_metric import RunningMetric
import os


from MY_CrossOver import _MY_UniformCrossover
from MY_Mutation import _MY_Mutation
from MY_Mutation import _MY_Mutation2
from MY_Sampling import _MY_SumFixSampling
from MY_Repair import _MY_Repair
from MY_Profile import Eval
from MY_Profile import set_parameters
from MY_Profile import save_ProfResult

from MY_Display import _MY_Display
from MY_Callback import _MY_Callback


_Ns={'Alex':8,
    'Google':13,
    'Mobile':28,
    'Squeeze':19,
    'Res50':18,}


class MyProblem(ElementwiseProblem):
    n=0
    n_eval=0
    #fp=0
    def __init__(self,_n):
        n=_n
        print("Initialize the problem for graph with " + str(n) + " layers.")
        _xl=np.full(12,0)
        _xu=[5,7,4,3,3,3,3,3,n,n,n,n]
        super().__init__(n_var=len(_xu),
                         n_obj=2,
                         n_constr=1,
                         xl=np.array(_xl),
                         xu=np.array(_xu))

        

    def _evaluate(self, x, out, *args, **kwargs):
        OK=False
        while OK==False:
            try:
                R=Eval(x)
                if R == -1:
                    R={'Latency':10000,'FPS':0.1,'Power':100000}
                    break
                OK=True
            except Exception as e:
                print(e)
                print("There is a problem in profiling")
                fp=open('Problems.txt','a')
                fp.write(str(x)+'\n')
                R={'Latency':10000,'FPS':0.001,'Power':100000}
                OK=True
                #input("Please try to solve it and press to continue...")
        print(R)
        #1/FPS
        FPS = R['FPS']
        R_FPS = 1000/FPS

        #Energy
        try:
            Energy = R_FPS*R['Power']
        except:
            print(f'\n\n\n************ already evaluated has no power \n*************\n\n\n')
            R=Eval(x)
            Energy = R_FPS*R['Power']
        

        #Latency
        Target_Latency=1000
        Latency = R['Latency']
        G1=Latency-Target_Latency
        print(f'G1 is: {G1}')
        Max_RFPS=100
        Max_Energy=300000
        Norm=True
        if Norm:
            R_FPS=R_FPS/Max_RFPS
            Energy=Energy/Max_Energy
        print(f'Normalized R_FPS: {R_FPS}  Energy: {Energy}')
        out["F"] = [R_FPS, Energy]
        out["G"] = [G1]

    
    
    
    
    
    
    
    
    
    
    
    def _evaluate2(self, x, out, *args, **kwargs):
        
        self.n_eval=self.n_eval+1
        print('\n\n\n')
        print(f'{self.n_eval} --> {x}')
        Target_Latency=20
        Pwr=(x[0]+1)+1.4*(x[1]+1)+1.2*(x[2]+1)
        t=[]
        for i,p in enumerate(x[8:11]):
            t.append((10/(x[i]+1)) * p)

        t.append(x[11] * 3)
        t[0] = t[0] * 1.4
        print(t)
        R_FPS = max(t)
        #R_FPS = 1000/FPS
        Latency = sum(t)
        Energy = R_FPS * Pwr + t[-1]*3
        G1=Latency - Target_Latency
        
        print(f'R_FPS:{R_FPS}, Power:{Pwr}, Energy:{Energy}')
        print(f'Latency:{Latency}, G1:{G1}')
        out["F"] = [R_FPS, Energy]
        out["G"] = [G1]
        #input()


Trm=200

#https://pymoo.org/interface/termination.html
_termination = MultiObjectiveDefaultTermination(
    #x_tol=1e-8,
    #cv_tol=1e-6,
    f_tol=0.001,
    nth_gen=5,
    n_last=Trm,
    n_max_gen=Trm,
    n_max_evals=Trm*100,
)
#__termination = get_termination("n_gen", 50)

'''termination = DefaultMultiObjectiveTermination(
    xtol=1e-8,
    cvtol=1e-6,
    ftol=0.0025,
    period=30,
    n_max_gen=1000,
    n_max_evals=100000
)'''
        
def main(_graph='Alex'):
    #graph='Res50'
    graph=_graph
    stime=time.time_ns()
    problem = MyProblem(_Ns[graph])

    
    set_parameters(_Graph=graph)
    #algorithm = NSGA2(pop_size=100)
    algorithm = NSGA2(pop_size=100,
    sampling=_MY_SumFixSampling(),
    #selection=TournamentSelection(func_comp=binary_tournament),
    crossover=_MY_UniformCrossover(prob=0.5),
    #mutation=_MY_Mutation(ProbFreqMutation=1/3,ProbHostMutation=1/2,ProbOrderMutation=1/4,ProbPartsMutation=1/4),
    #mutation=_MY_Mutation(ProbFreqMutation=1/12,ProbHostMutation=1/12,ProbOrderMutation=1/12,ProbPartsMutation=1/12),
    mutation=_MY_Mutation2(),
    repair=_MY_Repair(),
    #survival=RankAndCrowdingSurvival(),
    #eliminate_duplicates=True,
    #n_offsprings=None,
    #display=MultiObjectiveDisplay(),
    )

    res = minimize(problem,
                algorithm,
                #('n_gen', 200),
                seed=1,
                verbose=True,
                return_least_infeasible=True,
                termination=_termination,
                save_history=True,
                display=_MY_Display(),
                callback=_MY_Callback(delta_gen=2,
                        n_plots=4,
                        #only_if_n_plots=True,
                        #do_close=False,
                        key_press=False,
                        do_show=False),
                )
    etime=time.time_ns()
    print(f'GA finished. Duration: {(etime-stime)/10**9} s')
    save_ProfResult()
    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()
    return res


if __name__ == "__main__":
    main()

