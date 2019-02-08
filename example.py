from pyml_experiments import Experiment
from pyml_experiments.writers import StdoutWriter,Sqlite3Writer,WriterWrapper
import logging

logging.basicConfig(level=logging.DEBUG)
name="Test Experiment"
arguments={"dataset":"mnist","learning_rate":0.01}
device_id={"mon_id":"Android","details":{"ou":3}}
writer = WriterWrapper(StdoutWriter(), Sqlite3Writer('test.db'))

with Experiment(arguments=arguments,writer=writer) as log:

    for i  in range(100):
        log.new_iteration()
        log.add_value("loss",i)
        log.info("Computing loss")
        log.push_scope("validation")
        log.add_value("accuracy",i/100.0)
        log.push_scope("for_test")
        log.add_value("error",1.0-i/100.0)
        log.add_value("accuracy",i/1000.0)
        log.pop_scope()
        log.pop_scope()
        if (i%10 == 0):
            log.push_scope("test")
            log.add_value("coucou",i*5)
            log.pop_scope()
    
    log.error("Oups!")
