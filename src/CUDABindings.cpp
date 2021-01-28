#include <torch/extension.h>

#include <pybind11/chrono.h>

#include <c10d/ProcessGroupNCCL.hpp>


namespace c10d {

namespace {

c10::intrusive_ptr<ProcessGroup> createProcessGroupNCCL(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::milliseconds& timeout) {
  ProcessGroupNCCL::Options options;
  options.isHighPriorityStream = false;
  options.opTimeout = timeout;
  return c10::make_instrusive<ProcessGroupNCCL>(store, rank, size, options);
}

} // namespacef

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupNCCL",
        &createProcessGroupNCCL,
        py::call_guard<py::gil_scoped_release>());
}


} // namespace c10d
