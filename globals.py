import os

def set_affinity_mask(affinity_mask = None):

    if affinity_mask is None:
        cpu_count = os.cpu_count()
        affinity_mask = set(range(cpu_count))

    os.sched_setaffinity(0, affinity_mask)


def set_affinity_list(affinity_list = None):

    if affinity_list is None: set_affinity_mask(None)
    else: set_affinity_mask(set(affinity_list))


def set_affinity_str(affinity_str = None):

    if affinity_str is None or affinity_str.isspace(): set_affinity_mask(None)
    aff = [int(alloc) for alloc in affinity_str.split(",")]
    set_affinity_list(aff)
