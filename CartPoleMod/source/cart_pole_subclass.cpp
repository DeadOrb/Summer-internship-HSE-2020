#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "structmember.h"
#include <math.h>
#include <iostream>
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>
#include <vector>

// 267-288 https://github.com/XuMuK1/hdirllib/blob/master/hdirllib/simulators/pythonSimulators.py

typedef struct {
    PyObject_HEAD
    double gravity;
    double masscart;
    double masspole;
    double total_mass;
    double length;
    double polemass_length;
    double force_mag;
    double tau;
    double theta_threshold_radians;
    double x_threshold;
} CartPoleSubclassObject;

static int CartPoleSubclass_init(CartPoleSubclassObject *self, PyObject *args, PyObject *kwds) {
    import_array();

    if (!PyArg_ParseTuple(args, "dddddddddd", &self->gravity, &self->masscart, &self->masspole, &self->total_mass,
            &self->length, &self->polemass_length, &self->force_mag, &self->tau, &self->theta_threshold_radians,
            &self->x_threshold)) {
        return -1;
    }

    return 0;
}

static PyObject * CartPoleStep(CartPoleSubclassObject *self, PyObject *args) {
    import_array();
    int action;
    double x, x_dot, theta, theta_dot;

    if (!PyArg_ParseTuple(args, "ddddi", &x, &x_dot, &theta, &theta_dot, &action)) {
        return NULL;
    }

    double force;
    if (action == 0) {
        force = self->force_mag;
    } else {
        force = -self->force_mag;
    }

    double costheta = cos(theta), sintheta = sin(theta);

    double temp = (force + self->polemass_length * theta_dot * theta_dot * sintheta) / self->total_mass;
    double thetaacc = (self->gravity * sintheta - costheta * temp) / (self->length * (4.0 / 3.0 - self->masspole *
            costheta * costheta / self->total_mass));
    double xacc = temp - self->polemass_length * thetaacc * costheta / self->total_mass;

    x = x + self->tau * x_dot;
    x_dot = x_dot + self->tau * xacc;
    theta = theta + self->tau * theta_dot;
    theta_dot = theta_dot + self->tau * thetaacc;

    return Py_BuildValue("dddd", x, x_dot, theta, theta_dot);
}

static PyMethodDef CartPoleSubclass_methods[] = {
        {"step", (PyCFunction) CartPoleStep, METH_VARARGS,
                "Make step for cart pole simulation"
        },
        {NULL}  /* Sentinel */
};

static PyTypeObject CartPoleSubclassType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "subclass.CartPoleSubclass",      /* tp_name */
        sizeof(CartPoleSubclassObject),            /* tp_basicsize */
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
        "Subclass for boosting Cart Pole task in python",               /* tp_doc */
        0,                              /* tp_traverse */
        0,                              /* tp_clear */
        0,                              /* tp_richcompare */
        0,                              /* tp_weaklistoffset */
        0,                              /* tp_iter */
        0,                              /* tp_iternext */
        CartPoleSubclass_methods,                              /* tp_methods */
        0,                              /* tp_members */
        0,                              /* tp_getset */
        0,                              /* tp_base */
        0,                              /* tp_dict */
        0,                              /* tp_descr_get */
        0,                              /* tp_descr_set */
        0,                              /* tp_dictoffset */
        (initproc) CartPoleSubclass_init,                              /* tp_init */
        0,                              /* tp_alloc */
        PyType_GenericNew,              /* tp_new */
};

static PyModuleDef cartpolesubmodule = {
        PyModuleDef_HEAD_INIT,
        "cartpolesubclass",
        "Module with subclass CartPole.",
        -1,
};

PyMODINIT_FUNC PyInit_CartPoleSubclass(void) {
    PyObject *m;
    if (PyType_Ready(&CartPoleSubclassType) < 0)
        return NULL;

    m = PyModule_Create(&cartpolesubmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&CartPoleSubclassType);
    if (PyModule_AddObject(m, "CartPoleSubclass", (PyObject *) &CartPoleSubclassType) < 0) {
        Py_DECREF(&CartPoleSubclassType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
