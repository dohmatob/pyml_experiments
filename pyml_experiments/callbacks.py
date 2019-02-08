"""
Synopsis: Utilities for logging Machine Learning experiments. Built on top of
          Ludovic Denauyer's `pyml_experiments` github repo
Author: Elvis Dohmatob
"""
import os
import io
import typing
import logging
from copy import deepcopy
import json

import sqlite3
import pandas as pd

import torch

from .Experiment import Experiment
from .writers import Sqlite3Writer


def pyml_db_to_pandas(db_path, tables=["logs", "experiments"],
                      merge_on="_id", args_dict=None, exclude_args=[]):
    """
    Convert database created by `pym_experiments` from sqlite3 to pandas

    Returns
    -------
    DataFrame
    """
    conn = sqlite3.connect(db_path)
    dfs = []
    if isinstance(tables, str):
        tables = [tables]
    for table_name in tables:
        df = pd.read_sql_query("select * from %s" % table_name , conn)
        dfs.append(df)
    if len(dfs) > 1:
        df = pd.merge(*dfs, on=merge_on)
    else:
        df = dfs[0]

    # filter data base and only keep experiment with given arguments
    if args_dict is not None:
        for option, value in args_dict.items():
            if len(df) == 0:
                break
            if option not in df.columns:
                continue
            if option in exclude_args:
                continue
            try:
                if value is None:
                    df = df.loc[df[option].isnull()]
                else:
                    if isinstance(value, str):
                        df = df.query("%s=='%s'" % (option, value))
                    else:
                        df = df.query("%s==%s" % (option, value))
            except pd.core.computation.ops.UndefinedVariableError as e:
                logging.error("UndefinedVariableError: %s" % e)
                logging.warn("Hint: Source code for model probably changed!")
                return
    return df


class PyMLCallback(object):
    """
    Log artifacts of experiment
    """
    def __init__(self, args_dict: typing.Dict=None, db_path: str="logs.db",
                 model=None, checkpoint_freq: int=100):
        self.args_dict = {} if args_dict is None else args_dict
        self.db_path = db_path
        self.writer = Sqlite3Writer(db_path, update_every=1)
        self.logger = Experiment(arguments=args_dict, writer=self.writer)
        self.iteration = 0
        self.checkpoint_freq = checkpoint_freq
        self.model = model

    def add_scalar(self, name, value, *args, **kwargs):
        parts = name.split("/")
        metric = parts[0]
        scope = "/".join(parts[1:])
        if hasattr(value, "item"):
            value = value.item()
        self.logger.push_scope(scope)
        self.logger.add_value(metric, value)
        self.logger.pop_scope()

    def new_iteration(self):
        self.model_checkpoint()
        self.logger.new_iteration()
        self.logger.add_value("iteration", self.iteration)
        self.iteration += 1

    def add_histogram(self, name, param, iteration):
        pass

    def model_checkpoint(self, model=None, end=False):
        """
        Persist current state of model into experiments' database
        """
        if model is None:
            model = self.model
        if model is None or self.iteration % self.checkpoint_freq:
            if not end:
                return
        logging.info("Serializing model checkpoint...")
        model_token = serialize_model(model, self.args_dict)
        self.logger.add_value("model", model_token)

        # make sure changes are committed
        if end:
            self.new_iteration()

        return model_token


def check_logger(logger, proc=0, **kwargs):
    pid = os.getpid()
    loggers = []
    args_dict = {}
    if logger is not None:
        if isinstance(logger, dict):
            args_dict = logger
        else:
            loggers.append(logger)
    db_path = args_dict.get("db_path", "logs.db")
    db_path = "%s.%d" % (db_path, pid)
    if args_dict is not None:
        args_dict = deepcopy(args_dict)
        args_dict["pid"] = pid
        args_dict["proc"] = proc
    logger = PyMLCallback(args_dict=args_dict, db_path=db_path, **kwargs)
    loggers.append(logger)
    return PackedLoggers(loggers)


class PackedLoggers(object):
    """
    Combine a bunch of loggers into 1.
    """
    def __init__(self, loggers):
        self.loggers = loggers

    def add_scalar(self, name, value, **kwargs):
        for logger in self.loggers:
            logger.add_scalar(name, value, **kwargs)

    def add_scalars(self, scalars_dict, new_iteration=False, **kwargs):
        for name, value in scalars_dict.items():
            for logger in self.loggers:
                if new_iteration and hasattr(logger, "new_iteration"):
                    logger.new_iteration()
                logger.add_scalar(name, value, **kwargs)

    def add_histogram(self, *args, **kwargs):
        for logger in self.loggers:
            if hasattr(logger, "add_histogram"):
                logger.add_histogram(*args, **kwargs)

    def add_embedding(self, model, val_data, **kwargs):
        out = model.forward(val_data)
        out = torch.cat((out.data, torch.ones(len(out), 1)), 1)

        for logger in self.loggers:
            if hasattr(logger, "add_embedding"):
                self.logger.add_embedding(
                    out, metadata=out.data, label_img=val_data.data.double(),
                    **kwargs)

    def new_iteration(self):
        for logger in self.loggers:
            if hasattr(logger, "new_iteration"):
                logger.new_iteration()

    def model_checkpoint(self, model, **kwargs):
        for logger in self.loggers:
            if hasattr(logger, "model_checkpoint"):
                logger.model_checkpoint(model, **kwargs)


def serialize_model(model, args_dict=None):
    try:
        copy = {}
        for param, val in args_dict.items():
            if isinstance(val, list):
                val = tuple(val)
            copy[param] = val
        key = hash(frozenset(copy.items()))
        ofile = "model%s.pkl" % key
        args_ofile = "model-args%s.json" % key
        with open(args_ofile, 'w') as fp:
            json.dump(args_dict, fp)
        device = model.device
        del model.device
        torch.save(model.state_dict(), ofile)
        model.device = device
        logging.info("Serialized model to disk: %s" % ofile)
        return ofile
    except:
        buff = io.BytesIO()
        torch.save(model, buff)
        return buff.getvalue()
