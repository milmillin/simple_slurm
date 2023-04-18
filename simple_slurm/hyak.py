from typing import Dict, Any, NamedTuple, Optional
import subprocess
import math

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


PARTITION_TIME_SCALE = {
    "gpu-a100": 0.5,  # a100 is roughly twice as fast
    "gpu-a40": 1.0,
    "gpu-2080ti": 1.2,
    "gpu-rtx6k": 1.2,
}

TIME_UNTIL_FREE = 24  # assume currently used gpu will be free in 24 hours


def find_multiple_allocations(
    n: int, estimated_hours: float, constraint: Constraint = Constraint()
) -> list[Allocation]:
    partitions = parse_hyakalloc()

    EVENT = tuple[float, Allocation]
    # list of time and which allocations will be free
    events: list[EVENT] = []
    for partition in partitions:
        time_scale = PARTITION_TIME_SCALE.get(partition.partition, 1.5)
        events.append(
            (
                time_scale * TIME_UNTIL_FREE,
                Allocation(
                    partition.account, partition.partition, partition.stat.used, -1.0
                ),
            )
        )

    # fill up free spaces
    current_time = 1.0
    allocations = []
    estimated_eta = 0.0
    while len(allocations) < n:
        while max(_compute_scores(partitions, constraint)) < TOLERABLE_SCORE:
            if len(events) == 0:
                raise ValueError("cannot satisfy constraint")
            current_time = events[0][0]
            upto = len(events)
            for i, (t, alloc) in enumerate(events):
                if t > current_time:
                    upto = i
                    break
                # free up alloc
                partitions = [partition.add(alloc) for partition in partitions]
            events = events[upto:]
        best_alloc = _find_best_allocation(partitions, constraint)
        allocations.append(best_alloc)
        partitions = [partition.subtract(best_alloc) for partition in partitions]
        time_scale = PARTITION_TIME_SCALE.get(best_alloc.partition, 1.5)
        eta = current_time + time_scale * estimated_hours
        events.append((eta, best_alloc))
        events = sorted(events, key=lambda x: x[0])
        estimated_eta = max(estimated_eta, eta)

    print(f"estimated_eta: {estimated_eta:.2f}")
    return allocations
