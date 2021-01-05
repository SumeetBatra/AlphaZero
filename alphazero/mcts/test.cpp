//
// Created by Sumeet Batra on 1/4/21.
//
#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int i, int j){
    return i + j;
}

PYBIND11_MODULE(main, m) {
    m.doc() = "example docstring";
    m.def("add", &add, "func that adds two nums");
}

