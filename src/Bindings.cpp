#include <torch/extension.h>

#include <c10d/ProcessGroupMPI.hpp>


namespace c10d {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupMPI",
        &ProcessGroupMPI::createProcessGroupMPI,
        py::call_guard<py::gil_scoped_release>());
}

} // namespace c10d
