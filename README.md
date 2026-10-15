# sui

Slurm GPU utilization display: one box per GPU per node (red = allocated,
labelled with owner; green = free; grey = node unavailable), plus CPU/memory
allocation and pending GPU demand.

## Install

```bash
uv tool install git+https://github.com/mhyatt000/sui
```

## Usage

```bash
sui              # print once
sui -w 5         # live full-screen view, refresh every 5s
sui -p gpu       # one partition
sui --json       # machine-readable
```
