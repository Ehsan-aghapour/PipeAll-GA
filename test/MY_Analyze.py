#python MY_Analyze
#python MY_Analyze 0 Res50 3(need GAResult_Res50.pkl) (first argument is new_run and second is thee name(postfix) of target cnn, 
# and last argument is number of clusters for clutering GA front face)
#from copyreg import pickle
#https://pymoo.org/getting_started/part_4.html
from __future__ import annotations
from cProfile import label
from re import L
import MY_GA as GA
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import Hypervolume
from pymoo.util.running_metric import RunningMetric
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pymoo.visualization.scatter import Scatter
from sklearn.cluster import KMeans
import MY_Profile
import math


GARes_pkl=Path('./').resolve()
N_Clusters=3
s_colors=['darkturquoise','greenyellow','orchid']




n_evals = []             # corresponding number of function evaluations\
hist_F = []              # the objective space values in each generation
hist_cv = []             # constraint violation in each generation
hist_cv_avg = []         # average constraint violation in the whole population

global res
def GARun(_graph='Alex'):
    global res
    try:
        res=GA.main(_graph)
        DumpRes()
    except:
        DumpRes()


def His():
    hist=res.history
    k=res.history
    global X
    global F
    X, F = res.opt.get("X", "F")
    for algo in hist:

        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)

        # retrieve the optimum from the algorithm
        opt = algo.opt

        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())

        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])

def An_CV():
    vals = hist_cv_avg

    k = np.where(np.array(vals) <= 0.0)[0].min()
    print(f"Whole population feasible in Generation {k} after {n_evals[k]} evaluations.")

    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, vals,  color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, vals,  facecolor="none", edgecolor='black', marker="p")
    plt.axvline(n_evals[k], color="red", label="All Feasible", linestyle="--")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Constraint Violation")
    plt.legend()
    plt.show()

def An_HV():
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    
    metric = Hypervolume(ref_point= np.array([1.1, 1.1]),
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=approx_ideal,
                     nadir=approx_nadir)

    hv = [metric.do(_F) for _F in hist_F]

    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.show()

def An_RM():
    running = RunningMetric(delta_gen=4,
                        n_plots=4,
                        only_if_n_plots=False,
                        #do_close=False,
                        key_press=False,
                        do_show=True)

    for algorithm in res.history:
        p=running.notify(algorithm)


def DumpRes():
    with open('GAResult.pkl','wb') as pkf:
        pickle.dump(res,pkf)

def LoadRes(Graph=''):
    global res, GARes_pkl
    
    if Graph:
        Graph='_'+Graph
    
    fdir=GARes_pkl / ('GAResult'+Graph+'.pkl')
    with open(fdir,'rb') as pkf:
        res=pickle.load(pkf)
    return res


def An_CNV():
    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt0 = np.array([e.opt[0].F[0] for e in res.history])
    opt1 = np.array([e.opt[0].F[1] for e in res.history])

    plt.title("Convergence")
    plt.plot(n_evals, opt0, "--")
    #plt.yscale("log")
    plt.show()

    plt.plot(n_evals, opt1, "-")
    #plt.yscale("log")
    plt.show()
    #j=input("next...")

def plot_objects():
    #f=res.history[-1].reslut().F
    f=res.F
    # denormalization:
    ff=np.multiply(f,[100,300])

    #1
    #plot=Scatter()
    #plot.add(ff, facecolor="none", edgecolor="red")
    #plot.show()

    #2:
    plt.scatter(ff[:,0],ff[:,1],facecolors='none', edgecolors='r')
    plt.xlabel('Frame time (ms)')
    plt.ylabel('Energy (mj)')
    plt.show()

kkk=0
subscript_map = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
    "7": "₇", "8": "₈", "9": "₉", "a": "ₐ", "b": "♭", "c": "꜀", "d": "ᑯ",
    "e": "ₑ", "f": "բ", "g": "₉", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ",
    "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "q": "૧", "r": "ᵣ",
    "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "w": "w", "x": "ₓ", "y": "ᵧ",
    "z": "₂", "A": "ₐ", "B": "₈", "C": "C", "D": "D", "E": "ₑ", "F": "բ",
    "G": "G", "H": "ₕ", "I": "ᵢ", "J": "ⱼ", "K": "ₖ", "L": "ₗ", "M": "ₘ",
    "N": "ₙ", "O": "ₒ", "P": "ₚ", "Q": "Q", "R": "ᵣ", "S": "ₛ", "T": "ₜ",
    "U": "ᵤ", "V": "ᵥ", "W": "w", "X": "ₓ", "Y": "ᵧ", "Z": "Z", "+": "₊",
    "-": "₋", "=": "₌", "(": "₍", ")": "₎"}
superscript_map = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ",
    "e": "ᵉ", "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ᶦ", "j": "ʲ", "k": "ᵏ",
    "l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ", "p": "ᵖ", "q": "۹", "r": "ʳ",
    "s": "ˢ", "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ",
    "z": "ᶻ", "A": "ᴬ", "B": "ᴮ", "C": "ᶜ", "D": "ᴰ", "E": "ᴱ", "F": "ᶠ",
    "G": "ᴳ", "H": "ᴴ", "I": "ᴵ", "J": "ᴶ", "K": "ᴷ", "L": "ᴸ", "M": "ᴹ",
    "N": "ᴺ", "O": "ᴼ", "P": "ᴾ", "Q": "Q", "R": "ᴿ", "S": "ˢ", "T": "ᵀ",
    "U": "ᵁ", "V": "ⱽ", "W": "ᵂ", "X": "ˣ", "Y": "ʸ", "Z": "ᶻ", "+": "⁺",
    "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾"}
def plot_clustering(Graph='Alex',k=3):
    global kkk,s_colors
    LoadRes(Graph)
    MY_Profile.set_parameters(_Graph=Graph,new_run=False)
    #c_ref=["green","red", "yellow", "blue"]
    
    c_ref = list(mcolors.TABLEAU_COLORS.values())  # Use a predefined colormap from matplotlib
    c_ref=s_colors
    c_ref_lighter=[mcolors.to_rgba(color, alpha=0.4) for color in c_ref]
    
    c_ref=np.array(c_ref)
    c_ref_lighter=np.array(c_ref_lighter)

    #sort normalized front
    sorted_F=res.F[np.argsort(res.F[:,0])]

    #cluster based on sorted normalized front
    
    model = KMeans(n_clusters=k,n_init=10).fit(res.F)
    

    #calc energy and time by denormalizing
    ff=np.multiply(res.F,[100,300])

    #index of sorted F based on first column (object) that is time
    sort_indexes=np.argsort(ff[:,0])

    #sorted ff
    sorted_ff=ff[sort_indexes]

    plt.figure(figsize=(8, 6))
    plt.scatter(ff[:,0], ff[:,1], c=c_ref[model.labels_.astype(int)])
    #plt.scatter(sorted_F[:,0], sorted_F[:,1], c=c_ref[model.labels_.astype(int)])
    plt.xlabel('Execution Time Per Frame (ms)')
    plt.ylabel('Energy Per Frame (mj)')
    plt.savefig(Graph+'_Time.jpg', dpi=1000)

    plt.figure(figsize=(8, 6))
    plt.scatter(1000/ff[:,0], ff[:,1], c=c_ref[model.labels_.astype(int)])
    plt.xlabel('Throughput (FPS)')
    plt.ylabel('Energy Per Frame (mj)')
    plt.savefig(Graph+'_FPS.jpg', dpi=1000)

    columns=['Index','Order','Host (G N)','F_L(MHz)', 'F_B (MHz)', 'F_G (MHz)','Time (ms)','Energy (mj)']
    columns2=['Index','X','Time (ms)','Energy (mj)']
    
    d=[]
    d2=[]
    colors=[]
    colors2=[]
    Xf=open(Graph+'_Xs.csv','w')
    for i in range(len(sort_indexes)):
        t=[]
        t2=[]
        t.append(i)
        
        t2.append(i)
        t2.append(res.X[sort_indexes[i]])
        
        chromos=res.X[sort_indexes[i]]
        
        t.append(MY_Profile.Decode(chromos)[0])
        t.append(MY_Profile.Decode(chromos)[1])
        #t.append(MY_Profile.Decode(chromos)[2])
        t.append(round(MY_Profile.Decode(chromos)[2][0]*0.000001,2))
        t.append(round(MY_Profile.Decode(chromos)[2][1]*0.000001,2))
        t.append(round(MY_Profile.Decode(chromos)[2][2]*0.000000001,2))
        t.append(round(ff[sort_indexes[i]][0],2))
        t.append(round(ff[sort_indexes[i]][1],2))
        #kkk=t
        #return
        d.append(t)
        '''jjj=t[:]
        jjj[3]=jjj[3]*0.000001
        jjj[3][2]=jjj[3][2]*0.001
        d.append(jjj)'''
        
        for e in t:
            Xf.write(str(e)+',')
        Xf.write(str(model.labels_.astype(int)[sort_indexes[i]])+',')
        Xf.write(str(res.X[sort_indexes[i]])+'\n')

        t2.append(round(ff[sort_indexes[i]][0],2))
        t2.append(round(ff[sort_indexes[i]][1],2))
        d2.append(t2)

        tt=[]
        for j in range(len(columns)):
            tt.append(c_ref_lighter[model.labels_.astype(int)[sort_indexes[i]]])
        colors.append(tt)

        tt2=[]
        for j in range(len(columns2)):
            tt2.append(c_ref_lighter[model.labels_.astype(int)[sort_indexes[i]]])
        colors2.append(tt2)
    Xf.close()
    
    plt.figure()
    ax=plt.table(cellText=d, cellColours=colors,colLabels=columns, loc='center',cellLoc='center')
    plt.axis('off')
    #ax.auto_set_font_size(True)
    ax.auto_set_font_size(False)
    ax.set_fontsize(7)
    ax.auto_set_column_width(col=list(range(len(columns))))
    plt.savefig(Graph+'_Table.jpg', dpi=1000)

    plt.figure()
    ax=plt.table(cellText=d2, cellColours=colors2,colLabels=columns2, loc='center',cellLoc='center')
    plt.axis('off')
    ax.auto_set_font_size(False)
    ax.set_fontsize(8)
    ax.auto_set_column_width(col=list(range(len(columns2))))
    plt.savefig(Graph+'_TableX.jpg', dpi=1000)

    plt.show()
plot_clustering()

'''
from sklearn.cluster import KMeans


1:
kmeans = KMeans(n_clusters= 3)
label = kmeans.fit_predict(ff)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(ff[label == i , 0] , ff[label == i , 1] , label = i)
plt.legend()
plt.show()


2:
model = KMeans(n_clusters=5).fit(X)
# Visualize it:
#plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))


colors = np.array(range(0,len(ff)*10,10))
plt.scatter(ff[:,0],ff[:,1],c=colors)

for i in range(len(ff)):
    plt.annotate(r.X[i],(ff[i][0],ff[i][1]))
'''


def test():
    global points
    global colors
    points=[]
    colors=[]
    g=0
    for a in res.history:
        for p in a.opt:
            if not any(np.array_equal(p.F,y) for y in points) and g:
                print(f'{p.F} not exist in {points} appear in generation {g}--{p.data["n_gen"]}')
                #input()
                points.append(p.F)
                #colors.append(p.data['n_gen'])
                colors.append(g)
        g=g+1

    points=np.array(points)
    colors=np.array(colors)

    global last_points
    global last_colors
    last_points=[]
    last_colors=[]
    for k,p in enumerate(points):
        if any(np.array_equal(p,y) for y in res.F):
            last_points.append(p)
            last_colors.append(colors[k])
    last_points=np.array(last_points)
    last_colors=np.array(last_colors)

    '''for i in range(len(last_colors)):
        plt.scatter(last_points[:,0],last_points[:,1],c=last_colors)
        plt.annotate(last_colors[i],last_points[i])'''
    for i in range(len(colors)):
        plt.scatter(points[:,0],points[:,1],c=colors)
        plt.annotate(colors[i],points[i])

    sorted_F=res.F[np.argsort(res.F[:,0])]
    print(sorted_F[:,0])
    plt.plot(sorted_F[:,0],sorted_F[:,1],'-o')
    plt.show()


def plot_fronts(k=0,max=0,ignore=0):
    H=res.history
    global F1s
    global P_F1s
    F1s=[H[0].result().F]
    P_F1s=[H[0].result().opt]
    mapping={0:0}
    for i in range(1,len(H)):
        print(f'{H[i].result().F.shape} and {F1s[-1].shape} and {H[i].result().F.shape == F1s[-1].shape}')
        if not (H[i].result().F.shape == F1s[-1].shape):
            F1s.append(H[i].result().F)
            P_F1s.append(H[i].result().opt)
            mapping[len(F1s)-1]=i
        else:
            if not (H[i].result().F==F1s[-1]).all():
                F1s.append(H[i].result().F)
                P_F1s.append(H[i].result().opt)
                mapping[len(F1s)-1]=i
        
    
    for i in range(len(F1s)):
        P_F1s[i]=P_F1s[i][np.argsort(F1s[i][:,0])]
        F1s[i]=F1s[i][np.argsort(F1s[i][:,0])]
    
    _k=5
    if k:
        _k=k
    step=int(math.ceil((len(F1s)/_k)))
    print(f'step is {step}')
    #input()

    target_index=len(F1s)-1
    n=1
    while target_index>0:
        if n<=ignore:
            target_index=target_index-step
            n=n+1
            continue
        target_F=F1s[target_index]
        target_p=P_F1s[target_index]
        plt.plot(target_F[:,0],target_F[:,1],'-o',label=f'G:{mapping[target_index]} Points:{len(target_F)}')
        plt.legend()
        for i in range(len(target_F)):      
            plt.annotate(target_p[i].data['n_gen'],target_F[i])
        target_index=target_index-step
        if max and n>=max:
            break
        n=n+1
    
    plt.show()


# +
def main():
    global New_Run,Graph,res,N_Clusters
    if New_Run:
        
        
        GARun(Graph)
        His()
        An_CNV()
        An_CV()
        An_HV()
        An_RM()
    else:
        
        LoadRes(Graph)
        His() 
        An_CNV()
        An_CV()
        An_HV()
        #An_RM()
        
        plot_clustering(Graph,N_Clusters)
if __name__ == "__main__" and False:
    global New_Run,Graph,N_Clusters
    New_Run=1
    if len(sys.argv) >1:
        New_Run=int(sys.argv[1])
    if len(sys.argv) > 2:
        Graph=sys.argv[2]
    if len(sys.argv) > 3:
        N_Clusters=int(sys.argv[3])
        
def jupy():
    global New_Run,Graph
    New_Run=0
    Graph="Alex"
    main()
#jupy()
# -




# +
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def visualize_colors():
    # Get the list of color names
    colors = list(mcolors.cnames.keys())

    # Create a figure with subplots
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot color patches and labels
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i % 10, -(i // 10)), 1, 1, color=color))
        ax.text(i % 10 + 0.5, -(i // 10) + 0.5, color, ha='center', va='center', fontsize=10)

    # Set x-axis and y-axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(-len(colors) // 10 - 1, 0)

    # Set x-axis and y-axis labels
    ax.set_xlabel('Index')
    ax.set_ylabel('Color')

    # Set x-ticks and y-ticks
    ax.set_xticks(list(range(10)))
    ax.set_yticks([])

    # Adjust spacing between subplots
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.05, hspace=0.2)

    # Show the colors as patches with labels
    plt.show()
visualize_colors()

# -

s_colors=['deepskyblue','greenyellow','orchid']
def vis_selected_colors(selected_colors):
    
    # Create a figure to display the colors and their lighter versions
    fig, ax = plt.subplots(figsize=(8, 2))
    alphas=list(range(2,10,2))
    batch=len(alphas)+1
    # Plot color patches for selected colors and their lighter versions
    for i, color in enumerate(selected_colors):
        ax.add_patch(plt.Rectangle((i*batch, 0), 1, 1, color=color))
        ax.text(i*batch + 0.5, 0.5, color, ha='center', va='center', fontsize=10,rotation=90)
        for j,_alpha in enumerate(alphas):
            lighter_color = mcolors.to_rgba(color, alpha=_alpha/10)  # Generate lighter color with 60% opacity
            #print(color,lighter_color)
            ax.add_patch(plt.Rectangle((i*batch + j+1, 0), 1, 1, color=lighter_color))
            ax.text(i*batch+j+1 + 0.5, 0.5, str(_alpha), ha='center', va='center', fontsize=10)
    # Set x-axis and y-axis limits
    ax.set_xlim(0, len(selected_colors) * batch)
    ax.set_ylim(0, 1)

    # Set x-axis and y-axis labels
    ax.set_xlabel('Index')
    ax.set_ylabel('Color')

    # Set x-ticks and y-ticks
    ax.set_xticks(list(range(len(selected_colors) * batch)))
    ax.set_yticks([])

    # Show the colors and their lighter versions as patches
    plt.show()
vis_selected_colors(s_colors)

list(range(2,10,2))


def latex_color(names=['darkgreen','cyan'],opacity=0.6):
    for name in names:
        print(name)
        c=mcolors.to_rgba(name, alpha=opacity)
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=c))
        rgb=[int(cc*255) for cc in c[:3]]
        _opacity=c[3]*100
        print(f'Opacity is:{_opacity}')
        print(f'\definecolor{{str(rgb)[2:-2]}}{{{name}}}{{{rgb[0]},{rgb[1]},{rgb[2]}}}')

        ax.text( 0.5, 0.5, f'{rgb}:{_opacity}', ha='center', va='center', fontsize=10)
        plt.show()
cs=['darkgreen','cyan','lawngreen','greenyellow','pink','lightseagreen']
latex_color(names=cs)

c=mcolors.to_rgba('darkgreen', alpha=0.6)
textc=[int(cc*255) for cc in c]
type(textc)


