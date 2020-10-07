#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "structmember.h"
#include <math.h>
#include <vector>


typedef struct {
    PyObject_HEAD
    PyObject *mod_clip, *clip;
    double g, m, l, dt, max_torque, max_speed;
} PendulumSubclassObject;

static int PendulumSubclass_init(PendulumSubclassObject *self, PyObject *args, PyObject *kwds) {
    if (!PyArg_ParseTuple(args, "Odddddd", &self->clip, &self->g, &self->m, &self->l, &self->dt,
            &self->max_torque, &self->max_speed)) {
        return -1;
    }
    return 0;
}

static PyObject * SubStep(PendulumSubclassObject *self, PyObject *args) {
    double u, th, thdot;

    if (!PyArg_ParseTuple(args, "ddd", &u, &th, &thdot)) {
        return NULL;
    }

    double costs = (fmod((u + M_PI), (2 * M_PI)) - M_PI) + .1 * thdot * thdot + .001 * (u * u);

    double newthdot = thdot + (-3 * self->g / (2 * self->l) * sin(th + M_PI) + 3 / (self->m *
            self->l * self->l) * u) * self->dt;
    double newth = th + newthdot * self->dt;

    PyObject *tmp = PyObject_CallObject(self->clip, Py_BuildValue("ddd", newthdot, -self->max_speed, self->max_speed));

    if (!PyArg_Parse(tmp, "d", &newthdot)) {
        return NULL;
    }

    return Py_BuildValue("ddd", newth, newthdot, costs);
}

static PyMethodDef PendulumSubclass_methods[] = {
        {"substep", (PyCFunction) SubStep, METH_VARARGS,
                "Make step"
        },
        {NULL}  /* Sentinel */
};

static PyTypeObject PendulumSubclassType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "subclass.PendulumSubclass",      /* tp_name */
        sizeof(PendulumSubclassObject),            /* tp_basicsize */
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
        "Subclass for boosting",               /* tp_doc */
        0,                              /* tp_traverse */
        0,                              /* tp_clear */
        0,                              /* tp_richcompare */
        0,                              /* tp_weaklistoffset */
        0,                              /* tp_iter */
        0,                              /* tp_iternext */
        PendulumSubclass_methods,                              /* tp_methods */
        0,                              /* tp_members */
        0,                              /* tp_getset */
        0,                              /* tp_base */
        0,                              /* tp_dict */
        0,                              /* tp_descr_get */
        0,                              /* tp_descr_set */
        0,                              /* tp_dictoffset */
        (initproc) PendulumSubclass_init,                              /* tp_init */
        0,                              /* tp_alloc */
        PyType_GenericNew,              /* tp_new */
};

static PyModuleDef pendulumsubmodule = {
        PyModuleDef_HEAD_INIT,
        "pendulumsubclass",
        "Module with subclass Pendulum.",
        -1,
};

PyMODINIT_FUNC PyInit_PendulumSubclass(void) {
    PyObject *m;
    if (PyType_Ready(&PendulumSubclassType) < 0)
        return NULL;

    m = PyModule_Create(&pendulumsubmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PendulumSubclassType);
    if (PyModule_AddObject(m, "PendulumSubclass", (PyObject *) &PendulumSubclassType) < 0) {
        Py_DECREF(&PendulumSubclassType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
