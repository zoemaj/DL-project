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
    colors = [
        [0.6350, 0.0780, 0.1840],
        [0.4940, 0.1840, 0.5560],
        [0, 0.4470, 0.7410],
        [0, 0.5, 0]
    ]
    labels=["[1024]","[1024,64]"]
    if len(accuracy_list)!=len(colors): #verification
        print("there should be 4 parameters different")
        return
    for df in accuracy_list:
        df.rename(columns={df.columns[0]:"steps"}, inplace=True)
        df.rename(columns={df.columns[1]:"accuracy"}, inplace=True)
    plt.hlines(0,xmin=0,xmax=0.1,color='white',label=r"$\bf{lr=1e-2:}$")
    for df, color, label in zip(accuracy_list[:2], colors[:2], labels):
        plt.plot(df['steps'], df['accuracy'], color=color, label=label)
    plt.hlines(0,xmin=0,xmax=0.1,color='white',label=r"$\bf{lr=1e-3:}$")
    for df, color, label in zip(accuracy_list[2:], colors[2:], labels):
        plt.plot(df['steps'], df['accuracy'], color=color, label=label)
    plt.xlabel('Step',fontsize=fs)
    plt.ylabel('Validation accuracy',fontsize=fs)
    plt.xticks(fontsize=fs)  # Set the font size for the x-axis labels
    plt.yticks(fontsize=fs)  # Set the font size for the y-axis labels
    #cut to have only y between 45 and 87:
    plt.ylim(45,87)
    # Dynamically generate legend based on the order of appearance
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [4,0, 1, 5, 2, 3]  #order of the labels
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],fontsize=fs2)

    plt.grid()
    plt.tight_layout()  # Adjust layout to prevent cutoff
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
    
    colors = [
        [0.6350, 0.0780, 0.1840],
        [0.4940, 0.1840, 0.5560],
        [0, 0.4470, 0.7410],
        [0, 0.5, 0]
    ]
    labels=["[1024]","[1024,64]"]
    if len(loss_list)!=len(colors): #verification
        print("there should be 4 parameters different")
    #plot the loss between
    for df in loss_list:
        df.rename(columns={df.columns[0]:"steps"}, inplace=True)
        df.rename(columns={df.columns[1]:"loss"}, inplace=True)
    labels=["[1024]","[1024,64]"]
    #plot an horizontal line white with label = '---lr=1e-2---'
    plt.hlines(0,xmin=0,xmax=0.1,color='white',label=r"$\bf{lr=1e-2:}$")
    for df, color, label in zip(loss_list[:2], colors[:2], labels):
        plt.plot(df['steps'], df['loss'], color=color, label=label)
    plt.hlines(0,xmin=0,xmax=0.1,color='white',label=r"$\bf{lr=1e-3:}$")
    for df, color, label in zip(loss_list[2:], colors[2:], labels):
        plt.plot(df['steps'], df['loss'], color=color, label=label)
    plt.xlabel('Step',fontsize=fs)
    plt.ylabel('Train Loss',fontsize=fs)
    plt.xticks(fontsize=fs)  # Set the font size for the x-axis labels
    plt.yticks(fontsize=fs)  # Set the font size for the y-axis labels
    # Dynamically generate legend based on the order of appearance
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [4,0, 1, 5, 2, 3]  # Adjust this based on your actual data
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],fontsize=fs2)

    plt.grid()
    plt.tight_layout()  # Adjust layout to prevent cutoff

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
        #with specific orders: file that end with 1e-2-1024, 1e-2, 1e-3-1024, 1e-3
        accuracy_list=[]
        for file in os.listdir(path_folder): 

            if file.startswith('accuracy') and file.endswith('1e-2-1024.csv'):
                accuracy_list.append(pd.read_csv(path_folder+'/'+file))
        for file in os.listdir(path_folder):
            if file.startswith('accuracy') and file.endswith('1e-2.csv'):
                accuracy_list.append(pd.read_csv(path_folder+'/'+file))
        for file in os.listdir(path_folder):
            if file.startswith('accuracy') and file.endswith('1e-3-1024.csv'):
                accuracy_list.append(pd.read_csv(path_folder+'/'+file))
        for file in os.listdir(path_folder):
            if file.startswith('accuracy') and file.endswith('1e-3.csv'):
                accuracy_list.append(pd.read_csv(path_folder+'/'+file))
            
        #plot the accuracy
        plot_accuracy(accuracy_list,path_folder+'/'+plots_name+'-accuracy.eps',list_visualisation[0])
    if list_application[1]:
        #load all files in folder path_folder that begin with 'loss-'
        loss_list=[]
        for file in os.listdir(path_folder):
            #load all files in folder path_folder that begin with 'loss-'
            #with specific orders: file that end with 1e-2-1024, 1e-2, 1e-3-1024, 1e-3
            if file.startswith('loss') and file.endswith('1e-2-1024.csv'):
                loss_list.append(pd.read_csv(path_folder+'/'+file))
            if file.startswith('loss') and file.endswith('1e-2.csv'):
                loss_list.append(pd.read_csv(path_folder+'/'+file))
            if file.startswith('loss') and file.endswith('1e-3-1024.csv'):
                loss_list.append(pd.read_csv(path_folder+'/'+file))
            if file.startswith('loss') and file.endswith('1e-3.csv'):
                loss_list.append(pd.read_csv(path_folder+'/'+file))

        #plot the loss
        plot_loss(loss_list,path_folder+'/'+plots_name+'-loss.eps',list_visualisation[1])

