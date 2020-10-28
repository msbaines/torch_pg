# torch_pg

## Description
torch_pg is a collection of PyTorch [Distributed ProcessGroup Backends](https://pytorch.org/docs/stable/distributed.html#backends).


## Supports

* tested on PyTorch 1.6.0 and NCCL 2.7.8

## Installation

```bash
MPI_HOME=/path/to/mpi NCCL_HOME=/path/to/nccl pip install -e .
```

## Examples
### MPI

```python
import torch.distributed as dist
import torch_pg

# Install or replace built-in MPI backend.
torch_pg.init_mpi()

dist.init_process_group("mpi")
```

### NCCL

```python
import torch.distributed as dist
import torch_pg

# Install or replace built-in NCCL backend.
torch_pg.init_nccl()

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"

dist.init_process_group("nccl", rank=rank, world_size=world_size)
```

## License

torch_pg is licensed under the [BSD-3-Clause License](LICENSE).
