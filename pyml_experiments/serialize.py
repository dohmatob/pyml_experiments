"""
Synopsis: Serialize for next-level stuff
Author: Elvis Dohmatob <gmdopp@gmail.com>
"""
import os
import tempfile
import glob
import codecs
import logging
import sqlite3

import numpy as np
import pandas as pd


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
            # XXX
            if option == "hidden_dims":
                value = "\\n".join(list(map(str, value)))
            try:
                if value is None:
                    df = df.loc[df[option].isnull()]
                else:
                    df = df.loc[df[option] == value]
            except pd.core.computation.ops.UndefinedVariableError as e:
                logging.error("UndefinedVariableError: %s" % e)
                logging.warn("Hint: Source code for model probably changed!")
                return
    return df


def unserialize_model(model_str):
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as tmp:
        model_path = tmp.name
        with open(model_path, 'wb') as fd:
            fd.write(model_str)
            model = torch.load(model_path)
            return model


def load_model_from_pyml_exp_db(args_dict, only_done=True, logs_dir="."):
    # merge database from log files
    logs_dir = os.path.abspath(logs_dir)
    dfs = []
    for pyml_exp_db_path in glob.glob(os.path.join(logs_dir, "log*.db*")):
        logging.info("Loading experiments db %s" % pyml_exp_db_path)
        try:
            df = pyml_db_to_pandas(pyml_exp_db_path, args_dict=args_dict,
                                   exclude_args=["num_bootstraps",
                                                 "scores_file",
                                                 "inference",
                                                 "alpha",  # XXX rm
                                                 "num_threads",
                                                 "disable_eval",
                                   ])
        except (sqlite3.DatabaseError, pd.io.sql.DatabaseError) as e:
            # database is probably empy
            logging.error("DatabaseError: %s" % e)
            continue
        dfs.append(df)
    if not dfs:
        return
    df = pd.concat(dfs)
    df.fillna(value=np.nan, inplace=True)

    # maybe run was aborted
    if only_done:
        df = df.query("_state == 'done'")

    # get model binary string
    if len(df)  == 0:
        return

    if not "model" in df.columns:
        logging.warn("Database %s doesn't have a 'model' column" % (
            pyml_exp_db_path))
        return

    df = df[~df.model.isnull()]  # only consider experiments which with chckpts
    if len(df) == 0:
        return

    model_token = df.loc[df._end_date == df._end_date.max()].model.tolist()[0]
    try:
        # XXX hack to remove quotes from string
        model_token = model_token[1:-1]
        logging.info("Loading saved model state from %s" % model_token)
        state_dict = torch.load(model_token)
        print(state_dict)
        return state_dict, model_token
    except:
        # load model from binary string
        model_str = codecs.escape_decode(model_token[1:-1])[0][1:]

        copy = {}
        for param, val in args_dict.items():
            if isinstance(val, list):
                val = tuple(val)
            copy[param] = val
        key = hash(frozenset(copy.items()))
        ofile = "model%s.pkl" % key
        return unserialize_model(model_str), ofile
