
python MY_Analyze.py 0 Squeeze 1
(Set font for MobileNet to 6, for Squeeze and Res50 to 7 and for Alex and Googleto 8)
Use last argument of #clusters respectively: 3,3,1,3,1 for Alex, Google, Mobile, Res50, and Squeeze respectively.

If you use new run (first argument) then it run the GA_main() for specified graph (as second argument) and GA_main in MY_GA.py file start running GA. For profiling it call Eval function in MY_Profile.py. In profiling first it try to open ProfileReslut.pkl in Profile directory which is a dictionary of profiled results for the graph (So do not forget to place the related file from data dir in there [Also later fix it so that based on graph name it read ProfileResult_GraphName.pkl]). By the way if the requested config (Design point) is already evaluated it is in this profile dictionary data and read it otherwise, if No_Board flag is set to 1 it return -1 (it is for when there is a problem in profiling in Resnet50) and if No_Board is not set then it try to profile the desing point on the board.

When the New Run(first argument) is 0, then it use the saved result history of GA algorithm which is GA_Result_GraphName.pkl in current directory. 

In plots you can find the generated plots when running GA with New_run.

When New Run is 0, it plot HV, convergence and tables of first pareto front design point and related objective values.


** Why in Squeezenet the convergence plot which shows when the constraint is met, is weird.
