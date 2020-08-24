### 1. Use of .json

In this repo, .json files are used to fully describe the machine learning pipelines. They
are then used as arguments for the python execution. This means that only the actual compute is done
in python, every process is set out in the minimal language that is .json.  

We encourage this methodology as it makes distributed computing easier but it especially
makes adding new alogirthms very straightforward. You should develop very little code but
instead directly integrate .json structures.


### 2. File structure

The model computation is describred in 4 parts, and one computation is equivalent
to one .json file.

The basic structure is the following:

```
myjson.json
{
hypermodel_pars
model_pars
compute_pars
data_pars
out_pars
}
```

Let's review the structure in more detail and give some templates.


## 2.1. Model parameters

```json
"model_pars": {
            "model_uri"     : "",
            "dim_channel"   : ,
            "kernel_height" : [],
            "dropout_rate"  : ,
            "num_class"     : 
        }
```


## 2.2. Data loading parameters

This is perhaps the most difficult part to find a common abstraction, as the nature of data_loader
in machine learning can vary greatly. We thus explained it in more detail:  

[dataloader.md](https://github.com/arita37/mlmodels/blob/dev/docs/DEV_docs/dataloader.md)


## 2.3. Computation parameters

```json
"compute_pars": {
            "learning_rate" : ,
            "epochs"        : ,
            "checkpointdir" : ""
        },
```


## 2.4. Output parameters

```json
"out_pars": {
            "path"          : "",
            "checkpointdir" : ""
        }
```


## 2.5. Hyperparameters

```json
"hypermodel_pars":   {
        "learning_rate": {"type": "log_uniform", "init": 0.01,  "range" : [0.001, 0.1] },
        "num_layers":    {"type": "int", "init": 2,  "range" :[2, 4] },
        "size":    {"type": "int", "init": 6,  "range" :[6, 6] },
        "output_size":    {"type": "int", "init": 6,  "range" : [6, 6] },
        "size_layer":    {"type" : "categorical", "value": [128, 256 ] },
        "timestep":      {"type" : "categorical", "value": [5] },
        "epoch":         {"type" : "categorical", "value": [2] }
    },
```


### 3. Usage

You can define .json parameters directly in your loading code to test if it works. If 
it does you can then automise the process.

```python
# import library
import mlmodels
model_uri    = "model_tf.1_lstm.py"
model_pars   =  {  "num_layers": 1, "size": ncol_input, "size_layer": 128, "output_size": ncol_output, "timestep": 4,}
data_pars    =  {"data_path": "/folder/myfile.csv"  , "data_type": "pandas" }
compute_pars =  { "learning_rate": 0.001, }
out_pars     =  { "path": "ztest_1lstm/", "model_path" : "ztest_1lstm/model/"}
save_pars = { "path" : "ztest_1lstm/model/" }
load_pars = { "path" : "ztest_1lstm/model/" }

#### Load Parameters and Train
from mlmodels.models import module_load
module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)      
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)

#### Inference
metrics_val   =  module.fit_metrics( model, sess, data_pars, compute_pars, out_pars) # get stats
ypred         = module.predict(model, sess,  data_pars, compute_pars, out_pars)     # predict pipeline
```

### 4. Other

You can also add variants to your model parameters through .json files. For example 
we give here a benchmark file for evaluating model performances:

```json
{
"metric_list": ["mean_absolute_error", "mean_squared_error",
                                       "median_absolute_error",  "r2_score"] 
}
```


### 5. Examples

**Test**
```json
"test": {

              "hypermodel_pars":   {
             "learning_rate": {"type": "log_uniform", "init": 0.01,  "range" : [0.001, 0.1] },
             "num_layers":    {"type": "int", "init": 2,  "range" :[2, 4] },
             "size":    {"type": "int", "init": 6,  "range" :[6, 6] },
             "output_size":    {"type": "int", "init": 6,  "range" : [6, 6] },

             "size_layer":    {"type" : "categorical", "value": [128, 256 ] },
             "timestep":      {"type" : "categorical", "value": [5] },
             "epoch":         {"type" : "categorical", "value": [2] }
           },

            "model_pars": {
                "learning_rate": 0.001,     
                "num_layers": 1,
                "size": 6,
                "size_layer": 128,
                "output_size": 6,
                "timestep": 4,
                "epoch": 2
            },

            "data_pars" :{
              "path"            : 
              "location_type"   :  "local/absolute/web",
              "data_type"   :   "text" / "recommender"  / "timeseries" /"image",
              "data_loader" :  "pandas",
              "data_preprocessor" : "mlmodels.model_keras.prepocess:process",
              "size" : [0,1,2],
              "output_size": [0, 6]              
            },


            "compute_pars": {
                "distributed": "mpi",
                "epoch": 10
            },
            "out_pars": {
                "out_path": "dataset/",
                "data_type": "pandas",
                "size": [0, 0, 6],
                "output_size": [0, 6]
            }
        },
    
        "prod": {
            "model_pars": {},
            "data_pars": {}
        }
```

**Resnet**
```json
"resnet101": {
        "hypermodel_pars": {
            "learning_rate": {"type": "log_uniform", "init": 0.01, "range": [0.001, 0.1 ] }
        },

        "model_pars": {
            "model_uri": "model_tch.torchhub.py",
            "repo_uri": "pytorch/vision",
            "model": "resnet101",
            "num_classes": 10,
            "pretrained": 1,
            "_comment": "0: False, 1: True",
            "num_layers": 5,
            "size": 6,
            "size_layer": 128,
            "output_size": 6,
            "timestep": 4,
            "epoch": 2
        },
        "data_pars": {
            "dataset"  : "mlmodels/preprocess/generic.py:NumpyDataset:CIFAR10",


            "tfdataset_train_samples":2000,
            "tfdataset_test_samples":500,
            "tfdataset_train_batch_size": 10,
            "tfdataset_test_batch_size": 10,
            "tfdataset_data_path": "dataset/vision/cifar10/",


            "dataset_train_file_name":"cifar10_train.npz",
            "dataset_test_file_name":"cifar10_test.npz",
            "dataset_features_key":"X",
            "dataset_classes_key":"y",            

            "transform_uri" : "mlmodels.preprocess.image:torch_transform_generic"





        },
        "compute_pars": {
            "distributed": "mpi",
            "max_batch_sample": 10,
            "epochs": 5,
            "learning_rate": 0.001
        },
        "out_pars": {
            "checkpointdir": "ztest/model_tch/torchhub/resnet101/checkpoints/",
            "path": "ztest/model_tch/torchhub/resnet101/"
        }
    },
```