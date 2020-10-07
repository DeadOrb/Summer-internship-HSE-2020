#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "structmember.h"
#include <math.h>
#include <iostream>
#include <vector>

typedef struct {
    PyObject_HEAD
    PyObject *step, *states, *actions, *log_probs, *rewards, *returnLogProbs, *expandActionDim, *returnRewards, *trajId, *policy, *gymStep, *state;
    int maxIterations;
} ComputeSamplesObject;

static void ComputeSamples_dealloc(ComputeSamplesObject *self) {
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static int ComputeSamples_init(ComputeSamplesObject *self, PyObject *args, PyObject *kwds) {
    if (!PyArg_ParseTuple(args, "OiOOOOOOOOOOO", &self->step, &self->maxIterations, &self->states, &self->actions,
            &self->log_probs, &self->rewards, &self->returnLogProbs, &self->expandActionDim, &self->returnRewards,
            &self->trajId, &self->policy, &self->gymStep, &self->state)) {
        return -1;
    }
    if (!PyCallable_Check(self->step)) {
        PyErr_SetString(PyExc_TypeError, "Parameter must be callable");
        return -1;
    }

    int iterId = 0, done = 0;
    while (iterId < self->maxIterations and done == 0) {
        PyObject *policy_res;
        policy_res = PyObject_CallFunctionObjArgs(self->step, self->states, self->actions, self->log_probs, self->rewards,
                                                  self->returnLogProbs, self->expandActionDim, self->returnRewards,
                                                  self->trajId, self->policy, self->gymStep, self->state, NULL);

        if (!PyArg_ParseTuple(policy_res, "pOOOO", &done, &self->states, &self->actions, &self->log_probs, &self->rewards)) {
            return -1;
        }
        iterId++;
    }
    return 0;
}

static PyObject * returnResult(ComputeSamplesObject *self, PyObject *args) {
    PyObject *result = Py_BuildValue("OOOOO", self->states, self->actions, self->log_probs, self->rewards, self->trajId);
    return result;
}

static PyMethodDef ComputeSamples_methods[] = {
        {"returnResult", (PyCFunction) returnResult, METH_VARARGS,
                "Make t steps of Euler method and return last step"
        },
        {NULL}  /* Sentinel */
};

static PyTypeObject ComputeSamplesType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "computesamples.ComputeSamples",      /* tp_name */
        sizeof(ComputeSamplesObject),            /* tp_basicsize */
        0,                              /* tp_itemsize */
        (destructor) ComputeSamples_dealloc,                              /* tp_dealloc */
//        0,
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
        "Compute samples method on C++",               /* tp_doc */
        0,                              /* tp_traverse */
        0,                              /* tp_clear */
        0,                              /* tp_richcompare */
        0,                              /* tp_weaklistoffset */
        0,                              /* tp_iter */
        0,                              /* tp_iternext */
        ComputeSamples_methods,                              /* tp_methods */
        0,                              /* tp_members */
        0,                              /* tp_getset */
        0,                              /* tp_base */
        0,                              /* tp_dict */
        0,                              /* tp_descr_get */
        0,                              /* tp_descr_set */
        0,                              /* tp_dictoffset */
        (initproc) ComputeSamples_init,                              /* tp_init */
        0,                              /* tp_alloc */
        PyType_GenericNew,              /* tp_new */
};

static PyModuleDef computesamplesmodule = {
        PyModuleDef_HEAD_INIT,
        "computesamples",
        "Module with computation samples.",
        METH_VARARGS,
};

PyMODINIT_FUNC PyInit_ComputeSamples(void) {
    PyObject *m;
    if (PyType_Ready(&ComputeSamplesType) < 0)
        return NULL;

    m = PyModule_Create(&computesamplesmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&ComputeSamplesType);
    if (PyModule_AddObject(m, "ComputeSamples", (PyObject *) &ComputeSamplesType) < 0) {
        Py_DECREF(&ComputeSamplesType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
