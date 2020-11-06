#pragma once

#include_next <c10d/Utils.hpp>

namespace c10d {

#if (TORCH_MAJOR <= 1) && (TORCH_MINOR <= 6)
// For alltoall split size sanity check
inline void checkSplitSizes(
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    int group_size) {
  if (split_sizes.size() == 0) {
    TORCH_CHECK(
        tensor.size(0) % group_size == 0,
        "Tensor's dim 0 does not divide equally across group size");
  } else {
    TORCH_CHECK(
        split_sizes.size() == group_size,
        "Number of tensor splits not equal to group size");
    int sum = std::accumulate(split_sizes.begin(), split_sizes.end(), 0);
    TORCH_CHECK(
        sum == tensor.size(0), "Split sizes doesn't match total dim 0 size");
  }
}

// Compute alltoall lengths and offsets, handling multi-dimension tensors
template <typename T>
size_t computeLengthsAndOffsets(
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    std::vector<T>* lengths,
    std::vector<T>* offsets) {
  size_t group_size = lengths->size();
  bool equal_splits = false;
  size_t dim0_size = tensor.size(0);
  size_t row_size = (dim0_size ? tensor.numel() / dim0_size : 1);
  size_t split_size = 0;
  size_t offset = 0;

  if (split_sizes.size() == 0) {
    equal_splits = true;
    split_size = tensor.size(0) / group_size;
  }
  for (int i = 0; i < group_size; i++) {
    size_t length = row_size * (equal_splits ? split_size : split_sizes[i]);
    TORCH_INTERNAL_ASSERT(
        length <= std::numeric_limits<int>::max() &&
            offset <= std::numeric_limits<int>::max(),
        "Length or offset larger than INT_MAX not supported");
    (*lengths)[i] = length;
    (*offsets)[i] = offset;
    offset += length;
  }
  return offset;
}

template <typename T>
size_t computeLengthsAndOffsets(
    const std::vector<at::Tensor>& tensors,
    std::vector<T>* lengths,
    std::vector<T>* offsets) {
  size_t group_size = lengths->size();
  size_t offset = 0;
  for (int i = 0; i < group_size; i++) {
    size_t length = tensors[i].numel();
    TORCH_INTERNAL_ASSERT(
        length <= std::numeric_limits<int>::max() &&
            offset <= std::numeric_limits<int>::max(),
        "Length or offset larger than INT_MAX not supported");
    (*lengths)[i] = length;
    (*offsets)[i] = offset;
    offset += length;
  }
  return offset;
}

#endif /* (TORCH_MAJOR <= 1) && (TORCH_MINOR <= 6) */

} // namespace c10d
