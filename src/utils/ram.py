from collections import deque
from itertools import chain
import numpy as np
import psutil
from sys import getsizeof, stderr
import tracemalloc

from guppy import hpy
import objgraph
import pandas as pd
from pympler import tracker

from src.utils.text_sty import num2text


def print_heap():
    h = hpy()
    heap = h.heap()
    print(heap)


def print_top10_files_allocating_most_mem():
    """Requires tracemalloc.start() before the computation payload."""
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("Top 10 files allocating most memory:")
    for stat in top_stats[:10]:
        print(stat)


def print_memory_block_summary():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('traceback')

    mem_block_count = np.sum([stat.count for stat in top_stats])
    mem_block_size = np.sum([stat.size for stat in top_stats])
    print(f"There are {len(top_stats)} allocations with a total of "
          f"{num2text(mem_block_count)} memory blocks ({sizeof_fmt(mem_block_size)}).")

    top_stats_sorted_by_count = list(reversed(sorted(top_stats, key=lambda x: x.count)))
    print("Top 10 allocations with highest mem block count:")
    for allocation in top_stats_sorted_by_count[:10]:
        print(f"{num2text(allocation.count)} memory blocks of size {sizeof_fmt(allocation.size)} "
              f"with traceback {allocation.traceback}")

    # print("The biggest allocation has the following properties:")
    # stat = top_stats[0]
    # print(f"{num2text(stat.count)} memory blocks of size {sizeof_fmt(stat.size)}.")
    # print("Allocation traceback:")
    # for line in stat.traceback.format():
    #     print(line)


def print_leaking_objects_count():
    leaking_objects = objgraph.get_leaking_objects()
    leak_size = np.sum([getsizeof(obj) for obj in leaking_objects])
    print(f"There are {len(leaking_objects)} leaking objects ({sizeof_fmt(leak_size)}).")
    # print(leaking_objects)


def print_pympler_ram_usage():
    mem = tracker.SummaryTracker()
    mem_summary = pd.DataFrame(mem.create_summary(), columns=["Type", "Count", "Total Size"])
    mem_summary["Size/Object"] = mem_summary["Total Size"] / mem_summary["Count"]

    print("Total RAM usage (by pympler tracker): %s" % sizeof_fmt(mem_summary["Total Size"].sum()))

    # Print top 10 by total size
    top_10_total_size = mem_summary.sort_values('Total Size', ascending=False).head(10)
    top_10_total_size["Count"] = top_10_total_size["Count"].map(num2text)
    top_10_total_size["Total Size"] = top_10_total_size["Total Size"].map(sizeof_fmt)
    top_10_total_size["Size/Object"] = top_10_total_size["Size/Object"].map(sizeof_fmt)
    print(top_10_total_size.to_string())

    # Print top 10 by size per object
    top_10_object_size = mem_summary.sort_values('Size/Object', ascending=False).head(10)
    top_10_object_size["Count"] = top_10_object_size["Count"].map(num2text)
    top_10_object_size["Total Size"] = top_10_object_size["Total Size"].map(sizeof_fmt)
    top_10_object_size["Size/Object"] = top_10_object_size["Size/Object"].map(sizeof_fmt)
    print(top_10_object_size.to_string())


def print_total_ram_usage():
    rss = psutil.Process().memory_info().rss
    vms = psutil.Process().memory_info().vms
    print("Total RAM usage (by psutil):")
    print(f"- RSS (Resident Set Size): {sizeof_fmt(rss)}, only memory allocated in RAM.")
    print(f"- VMS (Virtual Memory Size): {sizeof_fmt(vms)}, also includes swapped out memory on hard drive.")


def print_object_ram_usage(obj, var_name: str):
    class_name = obj.__class__.__name__
    size_bytes = getsizeof(obj)
    print("Object '%s' of class %s has size %s." % (var_name, class_name, sizeof_fmt(size_bytes)))


def print_variables_ram_usage(variables):
    var_sizes = [(name, total_size(value)) for name, value in list(variables.items())]
    var_sizes_sorted = sorted(var_sizes, key=lambda x: -x[1])
    for name, size in var_sizes_sorted[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024
    return "%.1f %s%s" % (num, 'Yi', suffix)


def total_size(o, handlers=dict(), verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
