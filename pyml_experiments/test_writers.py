from writers import Sqlite3Writer

def test_find_type_for_variable():
    assert Sqlite3Writer._find_sqlite_type_for_variable(1)    == "INTEGER"
    assert Sqlite3Writer._find_sqlite_type_for_variable(True) == "BOOLEAN"
    assert Sqlite3Writer._find_sqlite_type_for_variable(1.2)  == "REAL"
    assert Sqlite3Writer._find_sqlite_type_for_variable("ac") == "TEXT"

def test_to_sqlite_value():
    assert Sqlite3Writer._to_sqlite_value(None)  == "null"
    assert Sqlite3Writer._to_sqlite_value(1)     == "1"
    assert Sqlite3Writer._to_sqlite_value(True)  == "1"
    assert Sqlite3Writer._to_sqlite_value(False) == "0"
    assert Sqlite3Writer._to_sqlite_value(1.2)   == "1.2"
    assert Sqlite3Writer._to_sqlite_value(False) == "0"

