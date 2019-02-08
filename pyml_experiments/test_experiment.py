from Experiment import Experiment
import writers

def test_can_build_Experiment_without_writer():
    # don't pass any writer. The goal is to ensure that it doesn't throw
    Experiment(arguments= {"foo": "bar"})

def test_can_use_with_statement_to_flush():
    writer = MockWriter()

    # checking preconditions
    assert not writer.began, "The writer should be started by the Experiment"
    assert not writer.exited

    with Experiment(writer=writer) as e:
        assert writer.began, "The writer should have been started by the Experiment"
        assert not writer.exited

    assert writer.exited, "The Experiment should have closed the writer upon exiting the with-statement"


class MockWriter(writers.Writer):
    def __init__(self):
        writers.Writer.__init__(self)
        self.exited = False
        self.began  = False

    def begin(self, arguments):
        self.began = True

    def write(self, values):
        pass

    def exit(self):
        self.exited = True

    def error(self, msg):
        pass

