import cx_Freeze
from glob import glob

executables = [cx_Freeze.Executable("launch.py"), cx_Freeze.Executable("level_editor.py")]

include_files = [(g, g) for g in glob("levels/*/*")]

cx_Freeze.setup(name="Outsmart",
                options={"build_exe":{"packages":["pyglet", "numpy"],
                                      "includes":["outsmart"],
                                      "include_files":include_files
                                     }
                        },
                executables=executables
                )
