import argparse
import hypertunning_algo

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('path_folder',help='path to the folder containing the files to compare')
    parser.add_argument('list_application',help='list "[Bool,Bool]" of the applications to compare for respectively validation accuracy plot, train loss plot')
    parser.add_argument('plots_name',help='name of the plots to save')
    parser.add_argument('list_visualisation',help='list "[Bool,Bool]" of the visualizations to plot for respectively validation accuracy plot, train loss plot')
    args=parser.parse_args()

    hypertunning_algo.execute(path_folder=args.path_folder,list_application=args.list_application,plots_name=args.plots_name,list_visualisation=args.list_visualisation)
