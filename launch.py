#!/usr/bin/env python3
import sys
import outsmart as osmt
import importlib.machinery
import argparse

lvl_directory = "levels"


def import_lvl(name):
    # https://stackoverflow.com/questions/27381264/python-3-4-how-to-import-a-module-given-the-full-path
    importlib.machinery.SourceFileLoader(name, '/'.join([lvl_directory, name, name])+'.py').load_module()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple commandline launcher for Outsmart.")
    parser.add_argument("lvl_name", metavar="level", nargs='?', help="name of a level present in the levels/ directory", default="tutorial")
    parser.add_argument("--TTS_OTF", action="store_true", help="activate on-the-fly speach synthesis. (default OFF)")
    parser.add_argument("--TTS_EXE", action="store", default="festival", help="activate on-the-fly speach synthesis. (default OFF)")
    
    args = parser.parse_args()
    #osmt.set_TTS_generate(args.TTS_OTF, args.TTS_EXE)
    osmt.set_TTS_generate(True, "OSX-say")
    import_lvl(args.lvl_name)
