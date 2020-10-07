#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "structmember.h"
#include <math.h>
#include <iostream>
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>
#include <vector>


typedef struct {
    PyObject_HEAD
    double force;
    double max_speed;
    double min_position;
    double max_position;
    double gravity;
    PyObject *clip;
} MountainCarSubclassObject;

static int MountainCarSubclass_init(MountainCarSubclassObject *self, PyObject *args, PyObject *kwds) {
    import_array();

    if (!PyArg_ParseTuple(args, "dddddO", &self->force,
            &self->max_speed, &self->min_position, &self->max_position, &self->gravity, &self->clip)) {
        return -1;
    }

    return 0;
}

static PyObject * MountainCarStep(MountainCarSubclassObject *self, PyObject *args) {
    import_array();
    double position, velocity, action;

    if (!PyArg_ParseTuple(args, "ddd", &position, &velocity, &action)) {
        return NULL;
    }

    velocity += (action - 1) * self->force + cos((3 * position)) * (-self->gravity);
    PyObject *vel_new = PyObject_CallObject(self->clip, Py_BuildValue("ddd", velocity, -self->max_speed,
            self->max_speed));

    if (!PyArg_Parse(vel_new, "d", &velocity)) {
        return NULL;
    }

    position += velocity;

    PyObject *pos_new = PyObject_CallObject(self->clip, Py_BuildValue("ddd", position, -self->min_position,
            self->max_position));

    if (!PyArg_Parse(pos_new, "d", &position)) {
        return NULL;
    }

    return Py_BuildValue("dd", position, velocity);
}

static PyMethodDef MountainCarSubclass_methods[] = {
        {"step", (PyCFunction) MountainCarStep, METH_VARARGS,
                "Make step for cart pole simulation"
        },
        {NULL}  /* Sentinel */
};

static PyTypeObject MountainCarSubclassType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "subclass.MountainCarSubclass",      /* tp_name */
        sizeof(MountainCarSubclassObject),            /* tp_basicsize */
        0,                              /* tp_itemsize */
        0,                              /* tp_dealloc */
        0,                              /* tp_print */
        0,                              /* tp_getattr */
        0,                              /* tp_setattr */
        0,                              /* tp_reserved */
        0,                              /* tp_repr */
        0,                              /* tp_as_number */
        0,                              /* tp_as_sequence */
        0,                              /* tp_as_mapping */
        0,                              /* tp_hash  */
        0,                              /* tp_call */
        0,                              /* tp_str */
        0,                              /* tp_getattro */
        0,                              /* tp_setattro */
        0,                              /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,             /* tp_flags */
        "Subclass for boosting Mountain Car task in python",               /* tp_doc */
        0,                              /* tp_traverse */
        0,                              /* tp_clear */
        0,                              /* tp_richcompare */
        0,                              /* tp_weaklistoffset */
        0,                              /* tp_iter */
        0,                              /* tp_iternext */
        MountainCarSubclass_methods,                              /* tp_methods */
        0,                              /* tp_members */
        0,                              /* tp_getset */
        0,                              /* tp_base */
        0,                              /* tp_dict */
        0,                              /* tp_descr_get */
        0,                              /* tp_descr_set */
        0,                              /* tp_dictoffset */
        (initproc) MountainCarSubclass_init,                              /* tp_init */
        0,                              /* tp_alloc */
        PyType_GenericNew,              /* tp_new */
};

static PyModuleDef mountaincarsubmodule = {
        PyModuleDef_HEAD_INIT,
        "mountaincarsubclass",
        "Module with subclass CartPole.",
        -1,
};

PyMODINIT_FUNC PyInit_MountainCarSubclass(void) {
    PyObject *m;
    if (PyType_Ready(&MountainCarSubclassType) < 0)
        return NULL;

    m = PyModule_Create(&mountaincarsubmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&MountainCarSubclassType);
    if (PyModule_AddObject(m, "MountainCarSubclass", (PyObject *) &MountainCarSubclassType) < 0) {
        Py_DECREF(&MountainCarSubclassType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
