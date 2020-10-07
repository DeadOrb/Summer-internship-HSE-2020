from distutils.core import setup, Extension
setup(name="MCS", version="1.0",
      ext_modules=[Extension("MountainCarSubclass", ["main.cpp"])])
