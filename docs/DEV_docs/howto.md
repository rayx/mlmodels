
## How to find information ?
<details>

Github Issues :
   https://github.com/arita37/mlmodels/issues?q=is%3Aopen+is%3Aissue+label%3Adev-documentation

This Howto.md file.

</details>
<br/>


## How to install mlmodels ?
<details>
There are two types of installations for ```mlmodels```.
The first is a manual controlled installation, the second is an automatic shell installation.

### Manual installation
The manual installation is dependant on [requirements.txt](https://github.com/arita37/mlmodels/blob/dev/requirements.txt)
and other similar text files.

Preview:
```
pandas<1.0
scipy>=1.3.0
scikit-learn==0.21.2
numexpr>=2.6.8
```


```bash
Linux/MacOS
pip install numpy<=1.17.0
pip install -e .  -r requirements.txt
pip install   -r requirements_fake.txt

Windows (use WSL + Linux)
pip install numpy<=1.17.0
pip install torch==1..1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .  -r requirements_wi.txt
pip install   -r requirements_fake.txt
```



### Automatic installation
One can also use the [run_install.sh](https://github.com/arita37/mlmodels/blob/dev/run_install.sh) and other similar files
for an automatic installation.
</details>
<br/>



## How to check if mlmodels works ?
<details>
Basic testing can be done with command line tool ```ml_test```.

### test_fast_linux : Basic Import check
```ml_test --do test_fast_linux```

1. [YAML](https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_fast_linux.yml)
2. [RAW LOGS](https://github.com/arita37/mlmodels_store/tree/master/log_import)
3. [CLEAN LOGS](https://github.com/arita37/mlmodels_store/tree/master/error_list/) 

### test_cli : Command Line Testing
```ml_test --do test_cli```

1. [YAML](https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_cli.yml)
2. [RAW LOGS](https://github.com/arita37/mlmodels_store/tree/master/log_test_cli)
3. [CLEAN LOGS](https://github.com/arita37/mlmodels_store/tree/master/error_list/)

### test_dataloader : Test if dataloader works
```ml_test --do test_dataloader```

1. [YAML](https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_dataloader.yml)
2. [RAW LOGS](https://github.com/arita37/mlmodels_store/tree/master/log_dataloader)
3. [CLEAN LOGS](https://github.com/arita37/mlmodels_store/tree/master/error_list/)

### test_jupyter : Test if jupyter notebooks works
```ml_test --do test_jupyter```

1. [YAML](https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_jupyter.yml)
2. [RAW LOGS](https://github.com/arita37/mlmodels_store/tree/master/log_jupyter)
3. [CLEAN LOGS](https://github.com/arita37/mlmodels_store/tree/master/error_list/)

### test_benchmark : benchmark
```ml_test --do test_benchmark```

1. [YAML](https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_benchmark.yml)
2. [RAW LOGS](https://github.com/arita37/mlmodels_store/tree/master/log_benchmark)
3. [CLEAN LOGS](https://github.com/arita37/mlmodels_store/tree/master/error_list/)

### test_pull_request : PR 
```ml_test --do test_jupyter```

1. [YAML](https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_pull_request.yml)
2. [RAW LOGS](https://github.com/arita37/mlmodels_store/tree/master/log_pullrequest)
3. [CLEAN LOGS](https://github.com/arita37/mlmodels_store/tree/master/error_list/)


You can then run basic codes and models to verify correct installation and
work environment.

```
cd mlmodels
python optim.py
python model_tch/textcnn.py
python model_keras/textcnn.py
```
</details>
<br/>



## How to check if one model works ?
<details>
### Run Model
Run/Test newly added model on your local machine or on 
[Gitpod](https://gitpod.io/) or [Colab](https://colab.research.google.com/).


Example of Gitpod use:
```
source activate py36
cd mlmodels
python model_XXXX/yyyy.py  
```



### Check Your Test Runs
https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_custom_model
</details>
<br/>


## How to develop using Colab ?
<details>
https://github.com/arita37/mlmodels/issues/262
<br/>
</details>


## How to develop using Gitpod ?
<details>
https://github.com/arita37/mlmodels/issues/101
</details>
<br/>


## How to add  a model ?
<details>
https://github.com/arita37/mlmodels/blob/adata2/README_addmodel.md

To add new model fork the repo. Inside the mlmodels directory we have multiple
subdirectories named like model_keras, model_sklearn and so on the idea is to use
**model_** before the type of framework you want to use. Now once you have decided the 
frame work create appripriately named model file and config file as described in the read me 
doc [README_addmodel.md](docs\README_docs\README_addmodel.md). The same model structure 
and config allows us to do the testing of all the models easily.
</details>
<br/>


## How to use Command Line CLI ?
<details>

https://github.com/arita37/mlmodels/blob/adata2/README_usage_CLI.md


</details>
<br/>

## How the model configuration JSON works ?
<details>
Sample of model written in JSON is located here :
     https://github.com/arita37/mlmodels/tree/dev/mlmodels/dataset/json
   
    https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/README_usage.md

A model computation is describred in 4 parts:

```
myjson.json
{
model_pars
compute_pars
data_pars
out_pars
}
```


</details>
<br/>

## How dataloader works ?
<details>
https://github.com/arita37/mlmodels/blob/dev/docs/DEV_docs/dataloader.md
</details>
<br/>



## How to improve the test process ?
<details>
  
Automatic testing is enabled and results are described here :
    https://github.com/arita37/mlmodels/blob/adata2/README_testing.md

Code for testing all the repo is located here:

   https://github.com/arita37/mlmodels/blob/dev/mlmodels/ztest.py


</details>
<br/>




## How to check test log after commit ?
<details>
Once the model is added we can do testing on it with commands like this, where model_framework is a placeholder for your selected framework and model_file.json is the config file for your model.

```
ml_models --do fit     --config_file model_framework/model_file.json --config_mode "test" 
```
Here the fit method is tested, you can check the predict fucntionality of the model like this.
```
ml_models --do predict --config_file model_tf/1_lstm.json --config_mode "test"
```
But this is individual testing that we can do to debug our model when we find an error in automatic the test logs.

We have automated testing in our repo and the results are stored in here https://github.com/arita37/mlmodels_store We havemultiple level logs and they are put under different directories as you can see here, log folders have **logs_** at the start.
![Mlmodels Store](imgs/test_repo.PNG?raw=true "Mlmodels Store")
We can focus on the error_list directory to debug our testing errors. Inside the error_list directory we can find the logs of all test cases in directories named at the time they are created
![Error List](imgs/error_list.PNG?raw=true "Error List")
Inside we can see separate files for each test cases which will have the details of the errors.
![Error Logs](imgs/error_logs.PNG?raw=true "Error logs")
For example we can look at the errors for test cli cases named as list_log_test_cli_20200610.md
![Error](imgs/test_cli_error.PNG?raw=true "Error")
We see multiple erros and we can click on the traceback for error 1 which will take us to the line 421 of the log file.
![Error Line](imgs/error_line.PNG?raw=true "Error Line")
We can see that while running the test case at line 418 caused the error, and we can see the error. 
```
ml_models --do fit  --config_file dataset/json/benchmark_timeseries/gluonts_m4.json --config_mode "deepar" 
```
So we fix the erorr by launch the git pod and test the test case again and see it works correctly after that we can commit teh changes and submit the pull request.
</details>
<br/>

## How to debug the repo ?
<details>
  
  
</details>

<br/>

















