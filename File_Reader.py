#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:53:27 2019

@author: victor
"""

#%% Import libs

import pandas as pd
import os



#%% File Reader

class FileReader(object):
    """
    A simple tool to read csv into pandas DataFrame
    """
    def __init__(self, filepath, filenames):
        assert(isinstance(filepath,str)), "Path should be a string!"
        if isinstance(filenames,str):
            n = 1
        elif isinstance(filenames,list) or isinstance(filenames,tuple):
            n = len(filenames)
        else:
            raise Exception("Names should be string or list or tuple!")
        
        self._path = filepath
        self._names = filenames
        self._n = n
        
    def Read(self,
             indexCol=0,
             useCols=None,
             concate=True,
             date=True,
             colName=True,
             fill=None,
             drop=False):
        """
        indexCol specifies index column;
        concate specifies whether to concate data if multiple names;
        fill specifies how to fill N/A;
        date specifies whether use date as index data type;
        colName changes column names to file names;
        drop specifies whether drop n/a rows
        """
        n = self._n
        
        # if only one name
        if n==1:
            if isinstance(self._names,str):
                name = self._names
            else:
                name = self._names[0]
            file = os.path.join(self._path, f"{name}.csv")
            data = pd.read_csv(file, index_col=indexCol, usecols=useCols,
                               parse_dates=date)
            
            if colName and data.shape[1]==1:
                data.columns = [name]
                
            if fill:
                if isinstance(fill,str):
                    data.fillna(method=fill, axis=0, inplace=True)
                else:
                    data.fillna(value=fill, axis=0, inplace=True)
                
            if drop:
                data.dropna(axis=0, inplace=True)
                
            return data
        
        # if multiple names
        else:
            if concate:
                files = [os.path.join(self._path, f"{name}.csv") 
                         for name in self._names]
                data = pd.concat(
                       (pd.read_csv(f,index_col=indexCol,usecols=useCols,
                                    parse_dates=date) 
                        for f in files), 
                        axis=1, sort=False)
                
                if colName and data.shape[1]==n:
                    data.columns = self._names
                    
                if fill:
                    if isinstance(fill,str):
                        data.fillna(method=fill, axis=0, inplace=True)
                    else:
                        data.fillna(value=fill, axis=0, inplace=True)
                
                if drop:
                    data.dropna(axis=0, inplace=True)
 
                return data
            
            # if not concate, output a dict
            else:
                data = {}
                for name in self._names:
                    file = os.path.join(self._path, f"{name}.csv")
                    data[name] = pd.read_csv(file, index_col=indexCol,
                                             usecols=useCols,
                                             parse_dates=date)
                    
                    if colName and data[name].shape[1]==1:
                        data[name].columns = [name]
                        
                    if fill:
                        if isinstance(fill,str):
                            data[name].fillna(method=fill, axis=0, inplace=True)
                        else:
                            data[name].fillna(value=fill, axis=0, inplace=True)
                
                    if drop:
                        data[name].dropna(axis=0, inplace=True)
                    
                return data
        
        
        
#%% Test

if __name__=='__main__':
        
    names_equity = ['GS','JPM','HSBC','UBS',
                    'WMT','KO','PG','PM',
                    'NEE','DUK','D','SO',
                    'BA','CAT','GE','MMM',
                    'XOM','CVX','COP','SLB']
    
    names_sector = 'Sectors'
    
    names_bench = '^GSPC'
    
    path = '/Users/victor/Desktop/Efforts/Portfolio Constructor/Data/Equities'
    
    data = FileReader(path,names_equity).Read(
            concate=True, useCols=[0,5],fill='pad', drop=True)
    
    breakdowns = FileReader(path,names_sector).Read(concate=False)
    
    benchmark = FileReader(path,names_bench).Read(
                 concate=False, useCols=[0,5])
    
    

