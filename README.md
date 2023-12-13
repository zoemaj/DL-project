# DL-project
# Few Shot adding Break His to the Benchmark

## Adding Files

This table shows which files need to be added to the code infrastructure, as well as the folder where to store it. The tumor `datasets/tumour` folder doesn't exist yet and needs to be added to the infrastructure manually.

*`/conf/dataset/break_his.yaml`*
*`/datasets/tumors/break_his.py`* 
*`/datasets/tumors/utils.py`* 

Additionnaly here are new file usefull for graphic representation:

*`main_hypertunning_algo.py`*
*`hypertunning_algo.py`*     
*`main_comparison_algo.py`*
*`comparison_algo.py`*


## Downloading the dataset

The dataset needs to be dowloaded and unzipped manually, in a similar way as SwissProt. The download link can be found in the `break_his.py` file and also here: `https://drive.google.com/file/d/1FlHhtTXzKgQCjxn18j1b4T3Q5FhylIn4/view`. It should be unzipped and stored in the folder `/data/break_his`.

Please verify that the the data is loaded and unzipped correctly. The structure should be as follows:
`/data/break_his/breast/{tumour_type}/SOB/{class}/{patient_number}/{magnification}`.

Once the files and images are stored as described, the code can be run. The first time the code is run image embeddings are created using our backbone (conv_next). Since this is fairly time-consuming and only the image embeddings are need to run the code, the embeddings are saved in the folder `data/break_his/embeddings_name`.

## Changing magnification and pretrained CNN

As the overall code structure should not be changed we could not pass the magnification size or the embedding CNN name as an input argument when running an experiment. Consequently, the code needs to be modified minorly to choose the different magnification and which embedding CNN use. In the `datasets/tumour/break_his.py` file change lines 38 and 78 `self.magnification = {your_desired_magnification}` and `self.pretrained_model_name = {Name_embedding_CNN}` as respectively an integer value (40, 100, 200, 400) and string {'convnext_base','efficientnet_b0'}. 

## Running the code
The run is executed with run.py file. Here an example of how use it:
`python3 run.py exp.name=baseline_pp-efficientnet-100X-1e-2-1024 method=baseline_pp dataset=break_his`

## Hyperparameters tunning

For the same reason than before, we could not pass the learning rate or backbones as an input argument when running an experiment. We then need to change manually these parameters in the files 'conf/main.yaml' and 'conf/dataset/break_his.yaml'. The train loss, validation loss and test accuracy can then be found in wandb.

For each algorithm we can select 4 different configurations and register their train loss and val accuracy from wandb into the folder results. Please try to keep this configuration :
*Results*
    *MX* with M a magnification (40,100,200,400)
        *CNN* with CNN beeing conv_next or efficientnet_b0
            *algorithm* with algorithm beeing baseline, baseline_pp, maml, matchingnet or protonet
                -contain 4 files, the 4 configurations possible (lr=1.e-2 or 1.e-3 and backbones=[1024] or [1024,64])
            *algorithm_comparisons* 
                -contain each best configuration of each algorithm

To run validation loss plot or validation accuracy please run the following command:

`python3 main_hypertunning_algo.py {path_to_algorithm} '[Bool,Bool]' '[Bool,Bool]'`

such that path_to_algorithm determines where to find the files for this algorithm
such that the first list of Boolean (True/False) indicates if respectively we execute and save the accuracy and/or the loss curve(s)
such that the second list of Boolean (True/False) indicates if we want to visualise the accuracy and/or the loss curve(s)

Example: `python3 main_hypertunning_algo.py results/100X/conv_next/maml '[True,True]' conv_next '[True,False]'`

## Comparison of the algorithms
To run validation loss plot or validation accuracy please run the following command:

`python3 main_hypertunning_algo.py {path_to_comparison} '[Bool,Bool]' '[Bool,Bool]'`

such that path_to_algorithm determines where to find the files for this algorithm
such that the first list of Boolean (True/False) indicates if respectively we execute and save the accuracy and/or the loss curve(s)
such that the second list of Boolean (True/False) indicates if we want to visualise the accuracy and/or the loss curve(s)

Example: `python3 main_comparison_algo.py results/100X/conv_next/algorithm_comparisons '[True,True]' '[False,True]' '`
