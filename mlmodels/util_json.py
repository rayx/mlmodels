"""

Alll related to json dynamic parsing


"""# -*- coding: utf-8 -*-
import os
import re
import fnmatch

# import toml
from pathlib import Path
from jsoncomment import JsonComment ; json = JsonComment()

import importlib
from inspect import getmembers

from mlmodels.util import *
from mlmodels.util import path_norm


####################################################################################################
class to_namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

    def get(self, key):
        return self.__dict__.get(key)


def log(*s, n=0, m=0):
    sspace = "#" * n
    sjump = "\n" * m
    print("")
    print(sjump, sspace, *s, sspace, flush=True)


####################################################################################################
def os_package_root_path(filepath="", sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    import mlmodels, os, inspect 

    path = Path(inspect.getfile(mlmodels)).parent
    # print( path )

    # path = Path(os.path.realpath(filepath)).parent
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path


###################################################################################################
def params_json_load(path, config_mode="test", 
                     tlist= [ "model_pars", "data_pars", "compute_pars", "out_pars"] ):
    from jsoncomment import JsonComment ; json = JsonComment()
    pars = json.load(open(path, mode="r"))
    pars = pars[config_mode]

    ### HyperParam, model_pars, data_pars,
    list_pars = []
    for t in tlist :
        pdict = pars.get(t)
        if pdict:
            list_pars.append(pdict)
        else:
            log("error in json, cannot load ", t)

    return tuple(list_pars)

#########################################################################################
#########################################################################################
def load_function(package="mlmodels.util", name="path_norm"):
  import importlib
  return  getattr(importlib.import_module(package), name)



def load_function_uri(uri_name="path_norm"):
    """
    #load dynamically function from URI

    ###### Pandas CSV case : Custom MLMODELS One
    #"dataset"        : "mlmodels.preprocess.generic:pandasDataset"

    ###### External File processor :
    #"dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"


    """
    
    import importlib, sys
    from pathlib import Path
    pkg = uri_name.split(":")

    assert len(pkg) > 1, "  Missing :   in  uri_name module_name:function_or_class "
    package, name = pkg[0], pkg[1]
    
    try:
        #### Import from package mlmodels sub-folder
        return  getattr(importlib.import_module(package), name)

    except Exception as e1:
        try:
            ### Add Folder to Path and Load absoluate path module
            path_parent = str(Path(package).parent.parent.absolute())
            sys.path.append(path_parent)
            #log(path_parent)

            #### import Absolute Path model_tf.1_lstm
            model_name   = Path(package).stem  # remove .py
            package_name = str(Path(package).parts[-2]) + "." + str(model_name)
            #log(package_name, model_name)
            return  getattr(importlib.import_module(package_name), name)

        except Exception as e2:
            raise NameError(f"Module {pkg} notfound, {e1}, {e2}")


def load_callable_from_uri(uri):
    assert(len(uri)>0 and ('::' in uri or '.' in uri))
    if '::' in uri:
        module_path, callable_name = uri.split('::')
    else:
        module_path, callable_name = uri.rsplit('.',1)
    if os.path.isfile(module_path):
        module_name = '.'.join(module_path.split('.')[:-1])
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)
    return dict(getmembers(module))[callable_name]
        

def load_callable_from_dict(function_dict, return_other_keys=False):
    function_dict = function_dict.copy()
    uri = function_dict.pop('uri')
    func = load_callable_from_uri(uri)
    try:
        assert(callable(func))
    except:
        raise TypeError(f'{func} is not callable')
    arg = function_dict.pop('arg', {})
    if not return_other_keys:
        return func, arg
    else:
        return func, arg, function_dict
    



def test_functions_json(arg=None):
  from mlmodels.util import load_function_uri

  path = path_norm("dataset/test_json/test_functions.json")
  dd   = json.load(open( path ))['test']
  
  for p in dd  :
     try :
         log("\n\n","#"*20, p)

         myfun = load_function_uri( p['uri'])
         log(myfun)

         w  = p.get('args', []) 
         kw = p.get('kw_args', {} )
         
         if len(kw) == 0 and len(w) == 0   : log( myfun())

         elif  len(kw) > 0 and len(w) > 0  : log( myfun( *w,  ** kw ))

         elif  len(kw) > 0 and len(w) == 0 : log( myfun( ** kw ))

         elif  len(kw) == 0 and len(w) > 0 : log( myfun( *w ))
                     
            
     except Exception as e:
        log(e, p )    



