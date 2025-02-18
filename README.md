# Simple Slurm (UW Hyak)

A fork from [Simple Slurm](https://github.com/amq92/simple_slurm) with added functionalities for UW Hyak.

```python
from simple_slurm import parse_hyakalloc, find_best_allocation, find_multiple_allocations, Constraints

# Allocate exact number of gpus and find optimal allocations for cpus and memory.
# It internally calls hyakalloc and prefers ones that are available now.
find_best_allocation(Constraint(gpus=1))
# Allocation(
#   account='realitylab',
#   partition='gpu-a100',
#   resources=Resources(cpus=6, memory=124, gpus=1)
# )

# You can also add minimum cpus and memory constraints.
find_best_allocation(Constraint(cpus=1, memory=8, gpus=1))

# Schedule n jobs each with the same estimated_time (in hr).
# It internally simulates when the resources are used and freed through time.
find_multiple_allocations(n=5, estimated_time=2, Constraint(gpus=1))
# estimated_eta: 3.00
# [
#   Allocation(...),
#   Allocation(...),
#   ...
# ]

# You can do your own calculation from the parsed hyakalloc.
parse_hyakalloc()
# [
#   Partition(
#     account='cse',
#     partition='gpu-2080ti',
#     stat=Stat(
#       free=Resources(cpus=4, memory=43, gpus=2),
#       used=Resources(cpus=36, memory=320, gpus=6),
#       total=Resources(cpus=40, memory=363, gpus=8)
#     )
#   ),
#   ...
# ]
```

## Installation

```
git clone git@github.com:milmillin/simple_slurm.git
cd simple_slurm
pip install -e .
```

<h1 align="center">Simple Slurm</h1>
<p align="center">A simple Python wrapper for Slurm with flexibility in mind<p>
<p align="center">
<a href="https://github.com/amq92/simple_slurm/actions/workflows/python-publish-pypi.yml">
    <img src="https://github.com/amq92/simple_slurm/actions/workflows/python-publish-pypi.yml/badge.svg" alt="Publish to PyPI" />
</a>
<a href="https://github.com/amq92/simple_slurm/actions/workflows/python-package-conda.yml">
    <img src="https://github.com/amq92/simple_slurm/actions/workflows/python-package-conda.yml/badge.svg" alt="Publish to Conda" />
</a>
<a href="https://github.com/amq92/simple_slurm/actions/workflows/python-run-tests.yml">
    <img src="https://github.com/amq92/simple_slurm/actions/workflows/python-run-tests.yml/badge.svg" alt="Run Python Tests" />
</a>
</p>

```python
import datetime

from simple_slurm import Slurm

slurm = Slurm(
    array=range(3, 12),
    cpus_per_task=15,
    dependency=dict(after=65541, afterok=34987),
    gres=['gpu:kepler:2', 'gpu:tesla:2', 'mps:400'],
    ignore_pbs=True,
    job_name='name',
    output=f'{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    time=datetime.timedelta(days=1, hours=2, minutes=3, seconds=4),
)
slurm.sbatch('python demo.py ' + Slurm.SLURM_ARRAY_TASK_ID)
```
The above snippet is equivalent to running the following command:

```bash
sbatch << EOF
#!/bin/sh

#SBATCH --array               3-11
#SBATCH --cpus-per-task       15
#SBATCH --dependency          after:65541,afterok:34987
#SBATCH --gres                gpu:kepler:2,gpu:tesla:2,mps:400
#SBATCH --ignore-pbs
#SBATCH --job-name            name
#SBATCH --output              %A_%a.out
#SBATCH --time                1-02:03:04

python demo.py \$SLURM_ARRAY_TASK_ID

EOF
```

## Contents
+ [Introduction](#introduction)
+ [Installation instructions](#installation-instructions)
+ [Many syntaxes available](#many-syntaxes-available)
    - [Using configuration files](#using-configuration-files)
    - [Using the command line](#using-the-command-line)
+ [Job dependencies](#job-dependencies)
+ [Additional features](#additional-features)
    - [Filename Patterns](#filename-patterns)
    - [Output Environment Variables](#output-environment-variables)



## Introduction

The [`sbatch`](https://slurm.schedmd.com/sbatch.html) and [`srun`](https://slurm.schedmd.com/srun.html) commands in [Slurm](https://slurm.schedmd.com/overview.html) allow submitting parallel jobs into a Linux cluster in the form of batch scripts that follow a certain structure.

The goal of this library is to provide a simple wrapper for these functions (`sbatch` and `srun`) so that Python code can be used for constructing and launching the aforementioned batch script.

Indeed, the generated batch script can be shown by printing the `Slurm` object:

```python
from simple_slurm import Slurm

slurm = Slurm(array=range(3, 12), job_name='name')
print(slurm)
```
```bash
>> #!/bin/sh
>> 
>> #SBATCH --array               3-11
>> #SBATCH --job-name            name
```

Then, the job can be launched with either command:
```python
slurm.srun('echo hello!')
slurm.sbatch('echo hello!')
```
```bash
>> Submitted batch job 34987
```

While both commands are quite similar, [`srun`](https://slurm.schedmd.com/srun.html) will wait for the job completion, while [`sbatch`](https://slurm.schedmd.com/sbatch.html) will launch and disconnect from the jobs.
> More information can be found in [Slurm's Quick Start Guide](https://slurm.schedmd.com/quickstart.html) and in [here](https://stackoverflow.com/questions/43767866/slurm-srun-vs-sbatch-and-their-parameters).

## Installation instructions

From PyPI

```bash
pip install simple_slurm
```

From Conda

```bash
conda install -c conda-forge simple_slurm
```

From git
```bash
pip install git+https://github.com/amq92/simple_slurm.git
```



## Many syntaxes available

```python
slurm = Slurm('-a', '3-11')
slurm = Slurm('--array', '3-11')
slurm = Slurm('array', '3-11')
slurm = Slurm(array='3-11')
slurm = Slurm(array=range(3, 12))
slurm.add_arguments(array=range(3, 12))
slurm.set_array(range(3, 12))
```

All these arguments are equivalent!
It's up to you to choose the one(s) that best suits you needs.

> *"With great flexibility comes great responsability"*

You can either keep a command-line-like syntax or a more Python-like one

```python
slurm = Slurm()
slurm.set_dependency('after:65541,afterok:34987')
slurm.set_dependency(['after:65541', 'afterok:34987'])
slurm.set_dependency(dict(after=65541, afterok=34987))
```

All the possible arguments have their own setter methods
(ex. `set_array`, `set_dependency`, `set_job_name`).

Please note that hyphenated arguments, such as `--job-name`, need to be underscored
(so to comply with Python syntax and be coherent).

```python
slurm = Slurm('--job_name', 'name')
slurm = Slurm(job_name='name')

# slurm = Slurm('--job-name', 'name')  # NOT VALID
# slurm = Slurm(job-name='name')       # NOT VALID
```

Moreover, boolean arguments such as `--contiguous`, `--ignore_pbs` or `--overcommit` 
can be activated with `True` or an empty string.

```python
slurm = Slurm('--contiguous', True)
slurm.add_arguments(ignore_pbs='')
slurm.set_wait(False)
print(slurm)
```
```bash
#!/bin/sh

#SBATCH --contiguous
#SBATCH --ignore-pbs
```




### Using configuration files

Let's define the *static* components of a job definition in a YAML file `default.slurm`

```yaml
cpus_per_task: 15
job_name: 'name'
output: '%A_%a.out'
```

Including these options with the using the `yaml` package is very *simple*

```python
import yaml

from simple_slurm import Slurm

slurm = Slurm(**yaml.load(open('default.slurm')))

...

slurm.set_array(range(NUMBER_OF_SIMULATIONS))
```

The job can be updated according to the *dynamic* project needs (ex. `NUMBER_OF_SIMULATIONS`).




### Using the command line

For simpler dispatch jobs, a comand line entry point is also made available.

```bash
simple_slurm [OPTIONS] "COMMAND_TO_RUN_WITH_SBATCH"
```

As such, both of these `python` and `bash` calls are equivalent.

```python
slurm = Slurm(partition='compute.p', output='slurm.log', ignore_pbs=True)
slurm.sbatch('echo \$HOSTNAME')
```
```bash
simple_slurm --partition=compute.p --output slurm.log --ignore_pbs "echo \$HOSTNAME"
```




## Job dependencies

The `sbatch` call prints a message if successful and returns the corresponding `job_id` 

```python
job_id = slurm.sbatch('python demo.py ' + Slurm.SLURM_ARRAY_TAKSK_ID)
```

If the job submission was successful, it prints:

```
Submitted batch job 34987
```

And returns the variable `job_id = 34987`, which can be used for setting dependencies on subsequent jobs

```python
slurm_after = Slurm(dependency=dict(afterok=job_id)))
```


## Additional features

For convenience, Filename Patterns and Output Environment Variables are available as attributes of the Simple Slurm object.

See [https://slurm.schedmd.com/sbatch.html](https://slurm.schedmd.com/sbatch.html#lbAH) for details on the commands.

```python
from slurm import Slurm

slurm = Slurm(output=('{}_{}.out'.format(
    Slurm.JOB_ARRAY_MASTER_ID,
    Slurm.JOB_ARRAY_ID))
slurm.sbatch('python demo.py ' + slurm.SLURM_ARRAY_JOB_ID)
```

This example would result in output files of the form `65541_15.out`.
Here the job submission ID is `65541`, and this output file corresponds to the submission number `15` in the job array. Moreover, this index is passed to the Python code `demo.py` as an argument.

> Note that they can be accessed either as `Slurm.<name>` or `slurm.<name>`, here `slurm` is an instance of the `Slurm` class.



### Filename Patterns

`sbatch` allows for a filename pattern to contain one or more replacement symbols.

They can be accessed with `Slurm.<name>`

name                | value | description
:-------------------|------:|:-----------
JOB_ARRAY_MASTER_ID | %A    |  job array's master job allocation number
JOB_ARRAY_ID        | %a    |  job array id (index) number
JOB_ID_STEP_ID      | %J    |  jobid.stepid of the running job. (e.g. "128.0")
JOB_ID              | %j    |  jobid of the running job
HOSTNAME            | %N    |  short hostname. this will create a separate io file per node
NODE_IDENTIFIER     | %n    |  node identifier relative to current job (e.g. "0" is the first node of the running job) this will create a separate io file per node
STEP_ID             | %s    |  stepid of the running job
TASK_IDENTIFIER     | %t    |  task identifier (rank) relative to current job. this will create a separate io file per task
USER_NAME           | %u    |  user name
JOB_NAME            | %x    |  job name
PERCENTAGE          | %%    |  the character "%"
DO_NOT_PROCESS      | \\\\  |  do not process any of the replacement symbols



### Output Environment Variables

The Slurm controller will set the following variables in the environment of the batch script.

They can be accessed with `Slurm.<name>`.

name                   | description
:----------------------|:-----------
SLURM_ARRAY_TASK_COUNT | total number of tasks in a job array
SLURM_ARRAY_TASK_ID    | job array id (index) number
SLURM_ARRAY_TASK_MAX   | job array's maximum id (index) number
SLURM_ARRAY_TASK_MIN   | job array's minimum id (index) number
SLURM_ARRAY_TASK_STEP  | job array's index step size
SLURM_ARRAY_JOB_ID     | job array's master job id number
...                    | ...

See [https://slurm.schedmd.com/sbatch.html](https://slurm.schedmd.com/sbatch.html#lbAK) for a complete list.
