#!/usr/bin/env python3
import sys
import importlib

lvl_directory = "levels"
lvl_name = "tutorial"


def import_lvl(name):
    importlib.import_module(".".join([lvl_directory, name, name]))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        lvl_name = sys.argv[1]
    import_lvl(lvl_name)
