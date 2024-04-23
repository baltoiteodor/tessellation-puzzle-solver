#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/stl.h>

namespace py = pybind11;

int add(int a, int b) {
    return a + b;
}

class DLXSolver{
public:
    DLXSolver(std::vector<std::vector<int>> rows, int width): _rows(rows), _width(width){};

    void printer() {
        std::cout << _rowsNum << " That s how many rows there are btw.\n";
    }

    std::vector<std::vector<int>> getRows() {
        return _rows;
    }

    int getWidth() {
        return _width;
    }
private:
    int _rowsNum = 0;
    std::vector<std::vector<int>> _rows;
    int _width = 0;
};

PYBIND11_MODULE(DLXCPP, handle) {
    handle.doc() = "Module DLX docs.";
    handle.def("add_cpp", &add);

    py::class_<DLXSolver>(
            handle, "DLXCPPSolver"
            )
        .def(py::init<std::vector<std::vector<int>>, int>())
        .def("printer", &DLXSolver::printer)
        .def("getRows", &DLXSolver::getRows)
        .def("getWidth", &DLXSolver::getWidth)
    ;
}