#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "structmember.h"
#include <math.h>
#include <vector>


typedef struct {
    PyObject_HEAD
    double power, min_action, max_action, max_speed, max_position, min_position;
} ContinuousMountainCarSubclassObject;

static int ContinuousMountainCarSubclass_init(ContinuousMountainCarSubclassObject *self, PyObject *args, PyObject *kwds) {
    if (!PyArg_ParseTuple(args, "dddddd", &self->power, &self->min_action, &self->max_action, &self->max_speed,
            &self->max_position, &self->min_position)) {
        return -1;
    }
    return 0;
}

static PyObject * Step(ContinuousMountainCarSubclassObject *self, PyObject *args) {
    double position, velocity, action;
    if (!PyArg_ParseTuple(args, "ddd", &position, &velocity, &action)) {
        return NULL;
    }

    double force = std::min(std::max(action, self->min_action), self->max_action);
    velocity += force * self->power - 0.0025 * cos(3 * position);

    if (velocity > self->max_speed) { velocity = self->max_speed; }
    if (velocity < -self->max_speed) { velocity = -self->max_speed; }

    position += velocity;

    if (position > self->max_position) { position = self->max_position; }
    if (position < self->min_position) { position = self->min_position; }
    if (position == self->min_position and velocity < 0) { velocity = 0; }

    return Py_BuildValue("dd", position, velocity);
}

static PyMethodDef ContinuousMountainCarSubclass_methods[] = {
        {"step", (PyCFunction) Step, METH_VARARGS,
                "Make step"
        },
        {NULL}  /* Sentinel */
};

static PyTypeObject ContinuousMountainCarSubclassType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "subclass.ContinuousMountainCarSubclass",      /* tp_name */
        sizeof(ContinuousMountainCarSubclassObject),            /* tp_basicsize */
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
        ContinuousMountainCarSubclass_methods,                              /* tp_methods */
        0,                              /* tp_members */
        0,                              /* tp_getset */
        0,                              /* tp_base */
        0,                              /* tp_dict */
        0,                              /* tp_descr_get */
        0,                              /* tp_descr_set */
        0,                              /* tp_dictoffset */
        (initproc) ContinuousMountainCarSubclass_init,                              /* tp_init */
        0,                              /* tp_alloc */
        PyType_GenericNew,              /* tp_new */
};

static PyModuleDef continuousmountaincarsubmodule = {
        PyModuleDef_HEAD_INIT,
        "mountaincarsubclass",
        "Module with subclass CartPole.",
        -1,
};

PyMODINIT_FUNC PyInit_ContinuousMountainCarSubclass(void) {
    PyObject *m;
    if (PyType_Ready(&ContinuousMountainCarSubclassType) < 0)
        return NULL;

    m = PyModule_Create(&continuousmountaincarsubmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&ContinuousMountainCarSubclassType);
    if (PyModule_AddObject(m, "ContinuousMountainCarSubclass", (PyObject *) &ContinuousMountainCarSubclassType) < 0) {
        Py_DECREF(&ContinuousMountainCarSubclassType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
