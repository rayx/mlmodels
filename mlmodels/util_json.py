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



import json
import os
import pandas as pd
import time


def os_folder_getfiles(folder):
    """

    :param folder: path of folder from which we want to extract all files paths
    :type folder: str
    :return: list of all files paths in folder
    :rtype: list of str
    """
    files_list = os.listdir(folder)
    all_files = list()
    for entry in files_list:
        full_path = os.path.join(folder, entry)
        if os.path.isdir(full_path):
            all_files = all_files + os_folder_getfiles(full_path)
        else:
            all_files.append(full_path)

    return all_files


def jsons_to_df(json_paths):
    """

    :param json_paths: list of json paths
    :type json_paths: list of str
    :return: DataFrame of the jsons
    :rtype: DataFrame
    """
    indexed_dicts = []
    problem = 0
    for i in range(len(json_paths)):
        try:
            with open(json_paths[i]) as json_file:
                d = dict()
                d['Path'] = json_paths[i]
                d['Json'] = json.load(json_file)
                indexed_dicts.append(d)
        except:
            if problem == 0:
                print("Files That have a structure problem:\n")
            problem += 1
            print('\t', json_paths[i])
            continue
    print("Total flawed jsons:\t", problem)
    all_jsons = []
    for i in range(len(indexed_dicts)):
        all_jsons.append(indexed_dicts[i]['Json'])
    paths = []
    filenames = []
    for i in range(len(indexed_dicts)):
        paths.append(indexed_dicts[i]['Path'])
    for i in range(len(indexed_dicts)):
        filename = indexed_dicts[i]['Path'].split('\\')[-1].split('.json')[0]
        filenames.append(filename)

    ddf = pd.json_normalize(all_jsons)
    df1 = pd.DataFrame({'file_path': paths, 'json_name': filenames})
    result = pd.concat([df1, ddf], axis=1)
    print("Dataframe created successfully")
    return result


def dict_update(fields_list, d, value):
    """

    :param fields_list: list of hierarchically sorted dictionary fields leading to value to be modified
    :type fields_list: list of str
    :param d: dictionary to be modified
    :type d: dict
    :param value: new value
    :type value: any type
    :return: updated dictionary
    :rtype: dict
    """
    if len(fields_list) > 1:
        l1 = fields_list[1:]
        k = fields_list[0]
        if k not in list(d.keys()):
            d[k] = dict()
        d[k] = dict_update(l1, d[k], value)
    else:
        k = fields_list[0]
        d[k] = value
        return d
    return d


def csv_to_json(csv):
    """

    :param csv: csv file (needs to be in current repository), containing jsons to be normalized
    :type csv: str
    :return: list of normalized jsons as dictionaries
    :rtype: list of dicts
    """
    dicts = []
    ddf = pd.read_csv(csv)
    fullname = list(ddf.columns)[3:]
    filename = list(ddf['json_name'])
    for i in range(len(filename)):
        dd = dict()
        fullname = list(ddf.columns)[3:]
        for j in range(len(fullname)):
            l = fullname[j].split('.')
            dd = dict_update(l, dd, None)
        for j in range(len(fullname)):
            value = ddf.iloc[i][fullname[j]]
            fields = fullname[j].split('.')
            dd.update(dict_update(fields, dd, value).copy())
        dicts.append(dd)

    paths = list(ddf['file_path'])
    dataset_dir = os_package_root_path()+'dataset'
    os.chdir(dataset_dir)
    paths = [p[len(dataset_dir)+1:] for p in paths]
    new_paths = []
    for i in range(len(paths)):
        lp = paths[i].split('\\')
        lp[0]='normalized_jsons'
        dire = '\\'.join(''.join(i) for i in lp[:len(lp)-1])
        new_paths.append(dire)
    for p in list(set(new_paths)):
        if not os.path.exists(p):
            os.makedirs(p)

    for i in range(len(filename)):
        with open(new_paths[i] + '\\' + filename[i] + '.json', 'w') as fp:
            json.dump(dicts[i], fp, indent=4)
    print("New normalized jsons created, check mlmodels\\mlmodels\\dataset")
    return dicts


def test_json_conversion():
    """
    Function to test converting jsons in dataset/json to normalized jsons
    :rtype: list of normalized jsons as dictionaries
    """
    json_folder_path = path_norm("dataset\\json")
    jsons_paths = os_folder_getfiles(json_folder_path)
    df = jsons_to_df(jsons_paths)
    df.to_csv('Jsons_Csv.csv')
    print('csv created successfully')
    time.sleep(1)
    New_dicts = csv_to_json('Jsons_Csv.csv')
    return New_dicts

# Testing code
if __name__ == "__main__":
    New_dicts = test_json_conversion()