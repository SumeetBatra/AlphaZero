//
// Created by Sumeet Batra on 1/5/21.
//
#include <boost/python.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(hello)
{
    class_<World>("World")
            .def("greet", &World::greet)
            .def("set", &World::set)
            ;
}