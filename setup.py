import cx_Freeze

executables = [cx_Freeze.Executable("launch.py")]

cx_Freeze.setup(name="Outsmart",
                options={"build_exe":{"packages":["pyglet", "numpy"],
                                      "include_files":[("levels/tutorial/tutorial.py", "levels/tutorial/tutorial.py"), ("levels/tutorial/tutorial.lab", "levels/tutorial/tutorial.lab"), ("levels/tutorial/tutorial.wild", "levels/tutorial/tutorial.wild"), "outsmart.py"]}
                        },
                executables=executables
                )
