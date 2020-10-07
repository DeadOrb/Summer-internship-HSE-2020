from distutils.core import setup, Extension
setup(name="CS", version="1.0",
      ext_modules=[Extension("ComputeSamples", ["main.cpp"])])
