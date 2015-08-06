import cx_Freeze
from glob import glob

executables = [cx_Freeze.Executable("launch.py"), cx_Freeze.Executable("levels/99level_editor/level_editor.py")]

include_files = [(g, g) for g in glob("levels/*/*")]
include_files += ["/Windows/System32/avbin64.dll", ("snd/", "snd/"), ("img/", "img/")]

cx_Freeze.setup(name="Outsmart",
                version="0.1",
                options={"build_exe":{"packages":["pyglet", "numpy"],
                                      "includes":["outsmart"],
                                      "include_files":include_files
                                     }
                        },
                executables=executables
                )
