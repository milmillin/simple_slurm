from typing import Dict, Any, NamedTuple, Optional, Union
import subprocess
import math
from datetime import datetime, timedelta
from queue import PriorityQueue

ACCOUNT_SCORE = {"cse": 1.0, "realitylab": 1.0, "stf": 1.0}

PARTITION_SCORE = {
    "gpu-a100": 0.99,
    "gpu-a40": 0.95,
    "gpu-2080ti": 0.8,
    "gpu-rtx6k": 0.8,
    "compute": 1.0,  # cpu only jobs would prefer these partitions
    "compute-hugemem": 1.0,
}

CPU_GPU_FALLOFF = 2  # allow 2 cpus per gpu less than average
MEM_GPU_FALLOFF = 16  # allow 16 GB per gpu less than average

DEFAULT_MIN_CPU = 1
DEFAULT_MIN_MEMORY = 8
DEFAULT_MIN_GPU = 0

TOLERABLE_SCORE = 1.25


def compute_logistic_constant(falloff: float, cutoff: float = 0.9) -> float:
    """Compute the logistic constant.

    Compute k such that for logistic function l(x) := 1 / (1 + exp(-kx),
    l(-falloff / 2) = 1 - cutoff and l(falloff / 2) = cutoff.

    Parameters
    ----------
    falloff : float
        The falloff of the logistic function.

    cutoff : float, optional
        The cutoff of the logistic function. Default is 0.9.

    Returns
    -------
    float
        The logistic constant.
    """
    return math.log((1 - cutoff) / cutoff) / -(falloff / 2)


def logistic(x: float, x0: float, k: float) -> float:
    """Compute the logistic function.

    Parameters
    ----------
    x : float
        The input.

    x0 : float
        The center of the logistic function.

    k : float
        The steepness of the logistic function.

    Returns
    -------
    float
        The output of the logistic function.
    """
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


CPU_GPU_K = compute_logistic_constant(CPU_GPU_FALLOFF)
MEM_GPU_K = compute_logistic_constant(MEM_GPU_FALLOFF)


def call_function(cmd: str) -> str:
    run = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    return run.stdout.decode("utf-8").strip()


class Resources(NamedTuple):
    cpus: int
    memory: int
    gpus: int

    def subtract(self, other: "Resources") -> "Resources":
        return Resources(
            self.cpus - other.cpus, self.memory - other.memory, self.gpus - other.gpus
        )

    def add(self, other: "Resources") -> "Resources":
        return Resources(
            self.cpus + other.cpus, self.memory + other.memory, self.gpus + other.gpus
        )


class Constraint(NamedTuple):
    cpus: int = DEFAULT_MIN_CPU
    memory: int = DEFAULT_MIN_MEMORY
    gpus: int = DEFAULT_MIN_GPU
    allowed_partitions: Optional[list[str]] = None


class Stat(NamedTuple):
    free: Resources
    used: Resources
    total: Resources

    def subtract(self, resources: Resources) -> "Stat":
        return Stat(self.free.subtract(resources), self.used.add(resources), self.total)

    def add(self, resources: Resources) -> "Stat":
        return Stat(self.free.add(resources), self.used.subtract(resources), self.total)


class Allocation(NamedTuple):
    account: str
    partition: str
    resources: Resources
    score: float


class Partition(NamedTuple):
    account: str
    partition: str
    stat: Stat

    def free_satisfies(self, constraint: Constraint) -> bool:
        if (
            constraint.allowed_partitions is not None
            and self.partition not in constraint.allowed_partitions
        ):
            return False
        return (
            self.stat.free.cpus >= constraint.cpus
            and self.stat.free.memory >= constraint.memory
            and self.stat.free.gpus >= constraint.gpus
        )

    def total_satisfies(self, constraint: Constraint) -> bool:
        if (
            constraint.allowed_partitions is not None
            and self.partition not in constraint.allowed_partitions
        ):
            return False
        return (
            self.stat.total.cpus >= constraint.cpus
            and self.stat.total.memory >= constraint.memory
            and self.stat.total.gpus >= constraint.gpus
        )

    def subtract(self, allocation: Allocation) -> "Partition":
        if (
            self.account == allocation.account
            and self.partition == allocation.partition
        ):
            return Partition(
                self.account, self.partition, self.stat.subtract(allocation.resources)
            )
        return self

    def add(self, allocation: Allocation) -> "Partition":
        if (
            self.account == allocation.account
            and self.partition == allocation.partition
        ):
            return Partition(
                self.account, self.partition, self.stat.add(allocation.resources)
            )
        return self

    def _key(self) -> tuple[str, str]:
        return (self.account, self.partition)


def parse_hyakalloc() -> list[Partition]:
    """
    Parse the output of hyakalloc.
    """
    run_result = call_function("hyakalloc")

    START_ROW = "╭"
    END_ROW = "╰"

    is_inside = False
    rows = []
    for row in run_result.split("\n"):
        if START_ROW in row:
            is_inside = True
        elif END_ROW in row:
            is_inside = False
            break
        else:
            if is_inside:
                rows.append(row)

    N_HEADER_ROWS = 1
    entries = [
        (rows[k], rows[k + 1], rows[k + 2])
        for k in range(N_HEADER_ROWS + 1, len(rows), 4)
    ]

    partitions = []
    for entry in entries:
        account = entry[0].split("│")[1].strip()
        partition = entry[0].split("│")[2].strip()
        total_cpus = int(entry[0].split("│")[3].strip())
        total_memory = int(entry[0].split("│")[4].strip()[:-1])
        total_gpus = int(entry[0].split("│")[5].strip())
        used_cpus = int(entry[1].split("│")[3].strip())
        used_memory = int(entry[1].split("│")[4].strip()[:-1])
        used_gpus = int(entry[1].split("│")[5].strip())
        free_cpus = int(entry[2].split("│")[3].strip())
        free_memory = int(entry[2].split("│")[4].strip()[:-1])
        free_gpus = int(entry[2].split("│")[5].strip())

        partitions.append(
            Partition(
                account=account,
                partition=partition,
                stat=Stat(
                    free=Resources(free_cpus, free_memory, free_gpus),
                    used=Resources(used_cpus, used_memory, used_gpus),
                    total=Resources(total_cpus, total_memory, total_gpus),
                ),
            )
        )

    return partitions


class QueueEntry(NamedTuple):
    account: str
    partition: str
    submit_time: datetime
    time_limit: timedelta
    end_time: Optional[datetime]
    state: str
    resources: Resources


def parse_time_limit(time_limit: str) -> timedelta:
    d = 0
    dhms = time_limit.split("-")
    if len(dhms) == 2:
        d, hms = dhms
    elif len(dhms) == 1:
        hms = dhms[0]
    else:
        raise ValueError(f"{time_limit} is not a valid time_limit string")
    h = 0
    m = 0
    s = 0
    hms = hms.split(":")
    s = hms[-1]
    if len(hms) > 1:
        m = hms[-2]
    if len(hms) > 2:
        h = hms[-3]
    return timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))


def parse_tres(tres: str) -> Resources:
    res = tres.split(",")
    res_dict: dict[str, str] = {k: v for k, v in map(lambda x: x.split("="), res)}
    mem = res_dict.get("mem", "0G")
    mem = int(mem[:-1]) if mem[-1] == "G" else 0
    return Resources(
        cpus=int(res_dict.get("cpu", 0)),
        memory=mem,
        gpus=int(res_dict.get("gres/gpu", 0)),
    )


def parse_squeue(accounts: Optional[list[str]]) -> list[QueueEntry]:
    acc = ""
    if accounts is not None:
        acc = " -A " + ",".join(accounts)
    output = call_function(
        f"squeue{acc} -h -t PD,R -O Account,Partition,SubmitTime,TimeLimit,EndTime,StateCompact,tres-alloc:70"
    )
    entries = output.split("\n")
    qentries = []
    for entry in entries:
        (
            account,
            partition,
            submit_time,
            time_limit,
            end_time,
            state,
            tres,
        ) = entry.split()

        qentries.append(
            QueueEntry(
                account=account,
                partition=partition,
                submit_time=datetime.strptime(submit_time, "%Y-%m-%dT%H:%M:%S"),
                time_limit=parse_time_limit(time_limit),
                end_time=datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
                if end_time != "N/A"
                else None,
                state=state,
                resources=parse_tres(tres),
            )
        )
    return qentries


def compute_score(partition: Partition, constraint: Constraint = Constraint()) -> float:
    """
    Compute a score for a given partition (higher is better).

    If a partition cannot satisfy the constraint even if it is free, returns -1.
    Else if it does not satisfy the constraint now, returns (0, 1]
    Else, returns (1, 2].

    If one or more gpus are requested, a partition with a score of >=TOLERABLE_SCORE
    will give you optimal resources within the defined FALLOFF.
    """

    if not partition.total_satisfies(constraint):
        return -1.0

    partition_score = PARTITION_SCORE.get(partition.partition, 0.5) * ACCOUNT_SCORE.get(
        partition.account, 0.5
    )

    if not partition.free_satisfies(constraint):
        return partition_score

    free = partition.stat.free
    total = partition.stat.total

    if constraint.gpus > 0:
        cpu_gpu_target = total.cpus // total.gpus
        mem_gpu_target = total.memory // total.gpus

        cpu_gpu_ratio = free.cpus // free.gpus
        mem_gpu_ratio = free.memory // free.gpus

        score = logistic(
            cpu_gpu_ratio, cpu_gpu_target - CPU_GPU_FALLOFF / 2, CPU_GPU_K
        ) * logistic(mem_gpu_ratio, mem_gpu_target - MEM_GPU_FALLOFF / 2, MEM_GPU_K)

        return (score - 0.5) * partition_score + 1.5
    else:
        return partition_score + 1.0


def _compute_scores(partitions: list[Partition], constraint: Constraint) -> list[float]:
    return [compute_score(partition, constraint) for partition in partitions]


def _find_best_allocation(
    partitions: list[Partition], constraint: Constraint
) -> Allocation:
    scores = _compute_scores(partitions, constraint)

    partition_score_pairs = sorted(
        zip(partitions, scores), key=lambda ps: ps[1], reverse=True
    )

    best_partition, best_score = partition_score_pairs[0]
    if best_score < 0.0:
        raise ValueError("No partition satisfies the given constraint")
    if constraint.gpus <= 0:
        # no gpus; you get what you requested
        return Allocation(
            best_partition.account,
            best_partition.partition,
            Resources(constraint.cpus, constraint.memory, gpus=0),
            best_score,
        )

    # with gpus; compute optimal cpu/mem usage
    free = best_partition.stat.free
    total = best_partition.stat.total
    optimal_cpus = max(constraint.gpus * (total.cpus // total.gpus), constraint.cpus)
    optimal_memory = max(
        constraint.gpus * (total.memory // total.gpus), constraint.memory
    )

    if best_score >= TOLERABLE_SCORE:
        # free now with acceptable tolerance; allocate min(free, optimal)
        optimal_cpus = min(free.cpus, optimal_cpus)
        optimal_memory = min(free.memory, optimal_memory)

    # otherwise, better wait for optimal allocation
    return Allocation(
        best_partition.account,
        best_partition.partition,
        Resources(optimal_cpus, optimal_memory, constraint.gpus),
        best_score,
    )


def find_best_allocation(constraint: Constraint = Constraint()) -> Allocation:
    partitions = parse_hyakalloc()
    return _find_best_allocation(partitions, constraint)


class _Event(NamedTuple):
    time: datetime
    alloc: Allocation


class _PendingTask(NamedTuple):
    time: datetime
    constraint: Constraint
    time_limit: timedelta


def find_multiple_allocations(
    n: int,
    estimated_time_limit: Union[
        float, timedelta, dict[str, float], dict[str, timedelta]
    ],
    constraint: Constraint = Constraint(),
) -> list[Allocation]:
    partitions = parse_hyakalloc()

    partition_map = {p._key(): p for p in partitions}

    pending_queues: dict[tuple[str, str], PriorityQueue] = {
        p._key(): PriorityQueue() for p in partitions
    }

    finishing_queue: PriorityQueue = PriorityQueue()

    avail_accounts = list(set([p.account for p in partitions]))

    entries = parse_squeue(avail_accounts)

    for entry in entries:
        key = (entry.account, entry.partition)
        if key not in partition_map:
            continue
        if entry.state == "R":
            assert entry.end_time is not None
            finishing_queue.put(
                _Event(
                    entry.end_time,
                    Allocation(entry.account, entry.partition, entry.resources, -1),
                )
            )
        else:
            cpus, mem, gpus = entry.resources
            pending_queues[key].put(
                _PendingTask(
                    entry.submit_time, Constraint(cpus, mem, gpus), entry.time_limit
                )
            )

    # fill up free spaces
    original_time = datetime.now()
    current_time = datetime.now()
    allocations = []
    estimated_eta = current_time
    while len(allocations) < n:
        while (
            max(_compute_scores(list(partition_map.values()), constraint))
            < TOLERABLE_SCORE
        ):
            if finishing_queue.empty():
                raise ValueError("cannot satisfy constraint")

            # fetch next finishing task
            finished = finishing_queue.get()
            assert isinstance(finished, _Event)
            current_time = finished.time

            # release resource
            key = (finished.alloc.account, finished.alloc.partition)
            new_partition = partition_map[key].add(finished.alloc)

            # see if pending can take it
            pending_queue = pending_queues[key]
            while not pending_queue.empty():
                next_pending_task = pending_queue.queue[0]
                assert isinstance(next_pending_task, _PendingTask)
                if new_partition.free_satisfies(next_pending_task.constraint):
                    # take it
                    cpus, mem, gpus, _ = next_pending_task.constraint
                    alloc = Allocation(key[0], key[1], Resources(cpus, mem, gpus), -1)
                    new_partition = new_partition.subtract(alloc)
                    pending_queue.get()
                    finishing_queue.put(
                        _Event(current_time + next_pending_task.time_limit, alloc)
                    )
                else:
                    break

            partition_map[key] = new_partition

        best_alloc = _find_best_allocation(list(partition_map.values()), constraint)
        allocations.append(best_alloc)

        key = (best_alloc.account, best_alloc.partition)
        partition_map[key] = partition_map[key].subtract(best_alloc)

        if isinstance(estimated_time_limit, dict):
            time_limit = estimated_time_limit[best_alloc.partition]
        else:
            time_limit = estimated_time_limit
        if not isinstance(time_limit, timedelta):
            time_limit = timedelta(hours=time_limit)

        eta = current_time + time_limit

        finishing_queue.put(_Event(eta, best_alloc))

        estimated_eta = max(estimated_eta, eta)

    eta_str = estimated_eta.strftime("%Y-%m-%d, %H:%M:%S")
    eta_delta_str = str(estimated_eta - original_time)
    print(f"estimated_eta: {eta_str} (in {eta_delta_str})")
    return allocations
