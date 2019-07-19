#include <Python.h>
#include <numpy/arrayobject.h>


static PyObject *speed_func(PyObject *self, PyObject *args) {
   /* Do your stuff here. */

   PyObject *arg1;
   PyObject *arg2;

   printf("Running program!");

   if (!PyArg_ParseTuple(args, "ii", &arg1, &arg2)) {
     return NULL;
   }

   return 3;
   /*
   PyObject *arg1;
   PyObject *arr1;
   int nd;
   printf("Created objects/pointers");

   if (!PyArg_ParseTuple(args, "O", &arg1)) {
     return NULL;
   }

   arr1 = PyArray_FROM_OTF(arg1, NPY_INT, NPY_IN_ARRAY);

   if (arr1 == NULL){
     return NULL;
   }

   return arr1;

   Py_DECREF(arr1);
   */
}

/*
static PyMethodDef speed_methods[] = {
   { "speed_func", (PyCFunction)speed_func, METH_NOARGS, NULL },
   { NULL, NULL, 0, NULL }
};
*/

static PyMethodDef SpeedMethods[] = {

    {"speed_func",  speed_func, METH_VARARGS,
     "return a string"},

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef speedmodule = {
    PyModuleDef_HEAD_INIT,
    "speed",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpeedMethods
};

PyMODINIT_FUNC
PyInit_speed(void)
{
    return PyModule_Create(&speedmodule);
}
