from distutils.core import setup, Extension
setup(name="PS", version="1.0",
      ext_modules=[Extension("PendulumSubclass", ["main.cpp"])])
