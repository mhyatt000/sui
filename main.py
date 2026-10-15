"""Per-node GPU utilization display for the Slurm cluster.

Each GPU is drawn as a box: red = allocated (labelled with the owner),
green = free, grey = node unavailable (down/drain). Utilization is *allocation*
as Slurm sees it, not live SM usage from nvidia-smi.

Cluster notes:
- AllocTRES does not track gres/gpu, so GPU usage comes from sinfo GresUsed,
  whose IDX list says exactly which GPU indices are taken.
- Per-GPU owners come from `scontrol show job -d`; untyped requests
  (gres/gpu:1, common on debug) show an empty GRES there, so those fall back
  to the squeue user list for the node.
- No JSON plugin or --delimiter, so sinfo uses the -O "Field:0|" suffix syntax.
"""

import argparse
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

GPU_RE = re.compile(r"gpu(?::([^:(,]+))?:(\d+)")
IDX_RE = re.compile(r"IDX:([^)]*)")

SINFO_FIELDS = [
    "NodeList", "Partition", "Gres", "GresUsed", "CPUsState",
    "AllocMem", "Memory", "CPUsLoad", "StateLong", "Reason",
]
UNAVAILABLE = ("down", "drain", "fail", "maint", "reboot", "not_responding", "future")


@dataclass
class Node:
    name: str
    partitions: list[str]
    gpu_type: str
    gpus_total: int
    used_idx: list[int]
    cpus_alloc: int
    cpus_total: int
    mem_alloc_mb: int
    mem_total_mb: int
    load: float
    state: str
    reason: str
    owners: dict[int, str] = field(default_factory=dict)  # gpu index -> user
    users: dict[str, int] = field(default_factory=dict)  # user -> GPUs held

    @property
    def gpus_used(self) -> int:
        return len(self.used_idx)

    @property
    def available(self) -> bool:
        return not any(s in self.state.lower() for s in UNAVAILABLE)

    @property
    def gpu_frac(self) -> float:
        return self.gpus_used / self.gpus_total if self.gpus_total else 0.0


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def run_lines(cmd: list[str]) -> list[str]:
    return [line for line in run(cmd).splitlines() if line.strip()]


def to_int(s: str) -> int:
    try:
        return int(float(s))
    except ValueError:
        return 0


def parse_gpus(gres: str) -> tuple[str, int]:
    """'gpu:l40s:4(S:0-1)' -> ('l40s', 4); sums multiple gpu entries."""
    types, total = [], 0
    for gtype, count in GPU_RE.findall(gres):
        total += int(count)
        if gtype and gtype not in types:
            types.append(gtype)
    return ",".join(types) or "gpu", total


def parse_idx(gres: str) -> list[int]:
    """'gpu:rtx_pro_6000:2(IDX:0,4)' -> [0, 4]; 'IDX:0-3' -> [0..3]; N/A -> []."""
    out: list[int] = []
    for spec in IDX_RE.findall(gres):
        for part in spec.split(","):
            if not part or part == "N/A":
                continue
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def expand_hostlist(hostlist: str) -> list[str]:
    if "[" not in hostlist:
        return hostlist.split(",")
    return run_lines(["scontrol", "show", "hostnames", hostlist])


def get_nodes() -> dict[str, Node]:
    fmt = ",".join(f"{f}:0|" for f in SINFO_FIELDS[:-1]) + f",{SINFO_FIELDS[-1]}:0"
    nodes: dict[str, Node] = {}
    for line in run_lines(["sinfo", "-N", "-h", "-O", fmt]):
        name, part, gres, used, cpus, amem, mem, load, state, reason = line.split("|", 9)
        part = part.rstrip("*")
        if name in nodes:  # node listed once per partition
            nodes[name].partitions.append(part)
            continue
        gpu_type, total = parse_gpus(gres)
        if total == 0:
            continue
        alloc, _idle, _other, cpu_total = cpus.split("/")
        reason = reason.strip()
        nodes[name] = Node(
            name=name, partitions=[part], gpu_type=gpu_type,
            gpus_total=total, used_idx=parse_idx(used),
            cpus_alloc=to_int(alloc), cpus_total=to_int(cpu_total),
            mem_alloc_mb=to_int(amem), mem_total_mb=to_int(mem),
            load=float(load) if load not in ("N/A", "") else 0.0,
            state=state, reason="" if reason == "none" else reason,
        )
    return nodes


def attach_owners(nodes: dict[str, Node]) -> None:
    """Fill per-GPU owners from scontrol, falling back to squeue per-node users."""
    for block in run(["scontrol", "show", "job", "-d"]).split("\n\n"):
        if "JobState=RUNNING" not in block:
            continue
        m = re.search(r"UserId=([^(\s]+)", block)
        user = m.group(1) if m else "?"
        for hosts, gres in re.findall(r"^\s+Nodes=(\S+) .*?GRES=(\S*)", block, re.M):
            for host in expand_hostlist(hosts):
                if host in nodes:
                    for i in parse_idx(gres):
                        nodes[host].owners[i] = user

    for line in run_lines(["squeue", "-h", "-t", "R", "-o", "%u|%N|%b"]):
        user, nodelist, tres = line.split("|")
        gpus = sum(int(n) for _, n in GPU_RE.findall(tres))
        if not gpus:
            continue
        for host in expand_hostlist(nodelist):
            if host in nodes:
                nodes[host].users[user] = nodes[host].users.get(user, 0) + gpus

    # Untyped jobs have no IDX in scontrol: hand unowned used GPUs to users
    # whose squeue GPU count isn't yet accounted for.
    for n in nodes.values():
        unowned = [i for i in n.used_idx if i not in n.owners]
        if not unowned:
            continue
        held = defaultdict(int)
        for u in n.owners.values():
            held[u] += 1
        for user, count in n.users.items():
            for _ in range(count - held[user]):
                if unowned:
                    n.owners[unowned.pop(0)] = user


def array_size(jobid: str) -> int:
    """'4178_[10-15%2]' -> 6 pending tasks; plain job id -> 1."""
    m = re.search(r"\[([^\]%]+)", jobid)
    if not m:
        return 1
    n = 0
    for part in m.group(1).split(","):
        lo, _, hi = part.partition("-")
        n += int(hi) - int(lo) + 1 if hi else 1
    return n


def get_pending() -> dict[str, dict[str, int]]:
    """partition -> {'jobs': n, 'gpus': n} for pending GPU jobs."""
    pend: dict[str, dict[str, int]] = defaultdict(lambda: {"jobs": 0, "gpus": 0})
    for line in run_lines(["squeue", "-h", "-t", "PD", "-o", "%i|%P|%b"]):
        jobid, parts, tres = line.split("|")
        gpus = sum(int(n) for _, n in GPU_RE.findall(tres))
        if not gpus:
            continue
        n = array_size(jobid)
        for part in parts.split(","):
            pend[part]["jobs"] += n
            pend[part]["gpus"] += n * gpus
    return dict(pend)


# ── rendering ────────────────────────────────────────────────────────────────

def gpu_boxes(n: Node) -> Text:
    t = Text()
    for i in range(n.gpus_total):
        if not n.available:
            t.append("  ✕  ", style="white on grey35")
        elif i in n.used_idx:
            t.append(f" {n.owners.get(i, '?')[:3]:<3} ", style="bold white on red3")
        else:
            t.append(f"  {i}  ", style="black on green3")
        t.append(" ")
    return t


def meter(used: float, total: float, width: int = 12) -> Text:
    frac = used / total if total else 0.0
    filled = round(frac * width)
    color = "green3" if frac < 0.5 else "yellow3" if frac < 0.9 else "red3"
    return (Text("█" * filled, style=color) + Text("░" * (width - filled), style="grey35")
            + Text(f" {frac:4.0%}", style="dim"))


def build_view(nodes: dict[str, Node], pending: dict[str, dict[str, int]]) -> Group:
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, title_justify="left",
                  title="[bold]GPU allocation[/]  [white on red3] user [/] used  "
                        "[black on green3] idx [/] free  [white on grey35]  ✕  [/] unavailable")
    table.add_column("Node", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("GPUs", no_wrap=True)
    table.add_column("Used", justify="right")
    table.add_column("CPU alloc", no_wrap=True)
    table.add_column("Mem alloc", no_wrap=True)
    table.add_column("Load", justify="right")
    table.add_column("State")

    for n in sorted(nodes.values(), key=lambda n: (n.partitions[0], n.name)):
        state = Text(n.state, style="green3" if n.state == "idle" else
                     "red3" if not n.available else "yellow3")
        if n.reason:
            state.append(f" {n.reason}", style="dim")
        table.add_row(
            f"{n.name} [dim]{','.join(n.partitions)}[/]", n.gpu_type, gpu_boxes(n),
            f"{n.gpus_used}/{n.gpus_total}",
            meter(n.cpus_alloc, n.cpus_total), meter(n.mem_alloc_mb, n.mem_total_mb),
            f"{n.load:.1f}", state,
        )

    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for n in nodes.values():
        if n.available:
            by_type[n.gpu_type][0] += n.gpus_used
            by_type[n.gpu_type][1] += n.gpus_total
    used = sum(u for u, _ in by_type.values())
    total = sum(t for _, t in by_type.values())

    footer = Text()
    footer.append("Cluster ", style="bold")
    footer.append_text(meter(used, total, width=24))
    footer.append(f"  {used}/{total} GPUs\n")
    footer.append("By type ", style="bold")
    footer.append("   ".join(f"{t} {u}/{tot} ({tot - u} free)" for t, (u, tot) in sorted(by_type.items())))
    if pending:
        footer.append("\nPending ", style="bold")
        footer.append("   ".join(f"{p}: {v['gpus']} GPUs / {v['jobs']} jobs"
                                 for p, v in sorted(pending.items())), style="yellow3")
    return Group(table, footer)


def snapshot(partition: str | None) -> tuple[dict[str, Node], dict[str, dict[str, int]]]:
    nodes = get_nodes()
    attach_owners(nodes)
    pending = get_pending()
    if partition:
        nodes = {k: v for k, v in nodes.items() if partition in v.partitions}
        pending = {k: v for k, v in pending.items() if k == partition}
    return nodes, pending


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-p", "--partition", help="only show nodes in this partition")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of the display")
    ap.add_argument("-w", "--watch", type=float, metavar="SEC", help="live refresh every SEC seconds")
    args = ap.parse_args()

    if args.json:
        nodes, pending = snapshot(args.partition)
        print(json.dumps({"nodes": [asdict(n) | {"gpus_used": n.gpus_used} for n in nodes.values()],
                          "pending": pending}, indent=2))
        return

    console = Console()
    if not args.watch:
        console.print(build_view(*snapshot(args.partition)))
        return

    import time
    with Live(build_view(*snapshot(args.partition)), console=console, screen=True) as live:
        try:
            while True:
                time.sleep(args.watch)
                live.update(build_view(*snapshot(args.partition)))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
