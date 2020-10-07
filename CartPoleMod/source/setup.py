from distutils.core import setup, Extension
setup(name="CPS", version="1.0",
      ext_modules=[Extension("CartPoleSubclass", ["cart_pole_subclass.cpp"])])
