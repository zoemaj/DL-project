import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
#define the size of the image
plt.rcParams["figure.figsize"] = (6,4)
fs=16
fs2=13

'------------------------------------------------------------------------------------------------'
def plot_accuracy(accuracy_list,accuracy_plot_name,visualisation=True):
    '''  
    plot the validation accuracy against the number of step for several algorithms
    input:
        accuracy_list: list of dataframes with the accuracy for each algorithm
        accuracy_plot_name: name of the plot saved
    '''
    names=['baseline_pp','baseline','matchingnet','maml','protonet'] #names of the algorithms
    if len(accuracy_list)!=len(names): #verification
        print("there should be only 5 different algorithms in the list: baseline_pp, baseline, matchingnet, maml, protonet")
        print("Please verify your list or implement a new name for the algorithm in comparison_algo.py file")
        return
    colors=['r','g','b','y','m']
    if len(accuracy_list)!=len(colors): #verification
        print("strange you don't have the same number of colors than algorithm names")
        return 
    for df in accuracy_list: #rename the headers of the dataframe to facilate the use after for the plot
        df.rename(columns={df.columns[0]:"steps"}, inplace=True)
        df.rename(columns={df.columns[1]:"accuracy"}, inplace=True)
    
    for df, name, color in zip(accuracy_list, names, colors):#plot the accuracy for each algorithm
        plt.plot(df['steps'], df['accuracy'], color=color, label=name)
    plt.xlabel('Step',fontsize=fs)
    plt.ylabel('Validation accuracy',fontsize=fs)
    plt.xlim(0,750) #limit the x axis to 750
    plt.xticks(fontsize=fs)  # Set the font size for the x-axis labels
    plt.yticks(fontsize=fs)  # Set the font size for the y-axis labels
    plt.legend(fontsize=fs2) # Set the font size for the legend
    plt.grid()
    plt.tight_layout()  # Adjust layout to prevent cutoff of the axis names
    #save the plot  as eps format
    plt.savefig(accuracy_plot_name, format='eps', dpi=1000)
    print("val. accuracy plot saved as:",   accuracy_plot_name )
    if visualisation==True:
        plt.show() 
    else:
        plt.close()
'------------------------------------------------------------------------------------------------'
def plot_loss(loss_list,loss_plot_name,visualisation=True):
    '''
    plot the train loss against the number of step for several algorithms
    input:
        loss_list: list of dataframes with the loss for each algorithm
        loss_plot_name: name of the plot saved
    '''
    names=['baseline_pp','baseline','matchingnet','maml','protonet'] #names of the algorithms
    if len(loss_list)!=len(names): #verification
        print("there should be only 5 different algorithms in the list: baseline_pp, baseline, matchingnet, maml, protonet")
        print("Please verify your list or implement a new name for the algorithm in comparison_algo.py file")
        return
    
    colors=['r','g','b','y','m']
    if len(loss_list)!=len(colors): #verification
        print("strange you don't have the same number of colors than algorithm names")
        return
    for df in loss_list: #rename the headers of the dataframe to facilate the use after for the plot
        df.rename(columns={df.columns[0]:"steps"}, inplace=True)
        df.rename(columns={df.columns[1]:"loss"}, inplace=True)
    
    fig, ax = plt.subplots(figsize=(6, 4)) #we create a subplot to insert a zoomed-in plot
    for df, name, color in zip(loss_list, names, colors): #plot the loss for each algorithm
        ax.plot(df['steps'], df['loss'], color=color, label=name)

    plt.xlabel('Step',fontsize=fs)
    plt.ylabel('Train Loss',fontsize=fs)
    plt.xticks(fontsize=fs)  # Set the font size for the x-axis labels
    plt.yticks(fontsize=fs)  # Set the font size for the y-axis labels
    plt.legend(fontsize=fs2) # Set the font size for the legend
    plt.grid()
    plt.tight_layout()  # Adjust layout to prevent cutoff of the axis names

    # Create a zoomed-in subplot for loss
    axins = ax.inset_axes([0.15, 0.3, 0.55, 0.55])  # (x, y, width, height)
    axins.set_xlim(0, 450)
    axins.set_ylim(0, 0.8)
    for df, name, color in zip(loss_list, names, colors): #plot the loss for each algorithm
        axins.plot(df['steps'], df['loss'], color, label=name)
    ax.indicate_inset_zoom(axins)

    #save the plot as eps
    plt.savefig(loss_plot_name, format='eps', dpi=1000)
    print("train loss plot saved as:",  loss_plot_name )
    if visualisation:
        plt.show()
    else: #doesnt show the plot
        plt.close()

'------------------------------------------------------------------------------------------------'
def execute(path_folder, list_application, plots_name, list_visualisation=[False,False]):
    '''
    execute the comparison between the different algorithms
    input:
        path_folder -> the folder where are the files to use
        list_application -> a list of 2 boolean indicating if we do the accuracy plot, the loss plot and the print accuracy. For example [True,False]
        list_visualisation -> a list of 2 boolean indicating if we want to see the plot or not. For example with [True,False] we will see the accuracy plot and not the loss plot
    '''
    #we first need to convert the list_application and list_visualisation as list becauuse we get them as string
    list_application=list_application.replace("[","").replace("]","").replace(" ","").split(",")
    list_visualisation=list_visualisation.replace("[","").replace("]","").replace(" ","").split(",")
    #convert the list_application as a list of boolean
    if list_application[0]=="True":
        list_application[0]=True
    else:
        list_application[0]=False
    if list_application[1]=="True":
        list_application[1]=True
    else:
        list_application[1]=False
    #convert the list_visualisation as a list of boolean
    if list_visualisation[0]=="True":
        list_visualisation[0]=True
    else:
        list_visualisation[0]=False
    if list_visualisation[1]=="True":
        list_visualisation[1]=True
    else:
        list_visualisation[1]=False

    #read the list_application as a list of boolean
    if list_application[0]:
        #load all files in folder path_folder that begin with 'accuracy-'
        accuracy_list=[]
        for file in os.listdir(path_folder):
            if file.startswith('accuracy-'):
                accuracy_list.append(pd.read_csv(path_folder+'/'+file))
        #plot the accuracy
        plot_accuracy(accuracy_list,path_folder+'/'+plots_name+'-accuracy.eps',list_visualisation[0])
    if list_application[1]:
        #load all files in folder path_folder that begin with 'loss-'
        loss_list=[]
        for file in os.listdir(path_folder):
            if file.startswith('loss-'):
                loss_list.append(pd.read_csv(path_folder+'/'+file))
        #plot the loss
        plot_loss(loss_list,path_folder+'/'+plots_name+'-loss.eps',list_visualisation[1])

