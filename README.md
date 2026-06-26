# Threaded vs Single-Threaded Training: Results and Analysis

Both programs train the exact same network (54 → 128 ReLU → 7 softmax) on the
same Covertype data, same seeds, same hyperparameters. The only variable
between them is whether the batch gradient computation runs on one thread or
is split across a thread pool with mutex-synchronized accumulation. Everything
below is measured on Vineeth's machine (12 logical threads), not estimated.

## 1. What's mechanically different

| | Single-threaded (`nn_single_threaded.cpp`) | Multi-threaded (`nn_multi_threaded.cpp`) |
|---|---|---|
| Gradient computation | One straight-line pass over the full batch | Batch split into 12 row-chunks, one `std::thread` per chunk |
| Shared state during compute | None needed, single thread owns everything | None touched during compute, each thread works on its own local `Gradients` object |
| Synchronization | None | A `std::mutex` is locked once per thread, only to add that thread's finished local result into a shared accumulator |
| Result | Same gradient sum, computed serially | Same gradient sum, computed in parallel, combined at the end |

This is data parallelism at the batch level: 12 threads each doing 1/12th of
the matrix math on a disjoint row range, with no thread touching another
thread's data, then a cheap final merge. It is the right model for this
problem because the per-row matrix multiplications have zero dependency on
each other until the final gradient sum.

## 2. Headline result

| | Single-threaded | Multi-threaded (12 threads) | Speedup |
|---|---|---|---|
| Single batch (4096 rows) | 51.49 ms | 12.18 ms | **4.23x** |
| Full training, 5 epochs | 26.75 s | 4.89 s | **5.47x** |

Same loss curve, same accuracy curve, in both runs: 1.62 → 0.99 loss, 49% →
63.5% test accuracy. The model learns identically either way, because the
math is identical either way. The only thing that changed is wall-clock time.

## 3. Why full-training speedup (5.47x) beat the microbenchmark speedup (4.23x)

This is the part worth understanding rather than just quoting. Look at the
per-epoch times:

| Epoch | Single-threaded | Multi-threaded |
|---|---|---|
| 0 | 3.22 s | 1.19 s |
| 1 | 5.92 s | 1.02 s |
| 2 | 5.73 s | 0.84 s |
| 3 | 5.90 s | 0.94 s |
| 4 | 5.98 s | 0.89 s |

The single-threaded run gets dramatically slower after epoch 0 (3.22s → a
steady ~5.88s average for epochs 1-4) while the multi-threaded run stays
roughly flat throughout. The most likely explanation is CPU frequency
behavior: sustaining 100% load on a single core for tens of seconds on a
laptop chip triggers thermal/power throttling, so that core's clock speed
drops and stays down. Spreading the same total work across 12 threads keeps
the per-core load lower, so the chip doesn't throttle the same way. This
isn't confirmed with a frequency profiler (`turbostat` or similar would
confirm it directly), but it's the standard explanation for "single-core
workload slows down over a sustained run, parallel workload doesn't." Worth
noting if asked, but not something to overclaim as fact without that
profiling data.

The practical upshot: the benefit of parallelizing this workload on real
laptop hardware is larger than a naive thread-count argument predicts, partly
because spreading load avoids triggering single-core throttling, on top of
the expected parallelism gain itself.

## 4. Why it's not a clean 12x

12 threads did not produce 12x speedup, and that's expected, not a flaw:

- **Hyperthreading**: "12 threads available" on a typical laptop chip usually
  means 6 physical cores with 2 logical threads each. Hyperthreaded logical
  cores share execution units on the same physical core, so they don't double
  real arithmetic throughput, only Amdahl's law applied to true parallel
  resources gets you the full count.
- **Memory bandwidth contention**: all 12 threads are reading from the same
  matrices in RAM. Once enough threads are active, they start competing for
  the same memory bus and cache lines, so adding threads stops scaling
  linearly well before the thread count is exhausted.
- **Thread spawn/join overhead**: every batch in this implementation creates
  12 new `std::thread` objects and joins them. For very small batches this
  overhead would dominate and the threaded version would actually be slower,
  it only pays off because each batch (4096 rows) gives each thread enough
  real work to amortize the spawn cost.
- **The serialized accumulation step**: the mutex lock at the end is brief, but
  it is a hard serialization point. 12 threads cannot all merge into the
  shared accumulator simultaneously, they queue for the lock one at a time.

A 4-5x measured speedup on a 12-thread machine, for this kind of workload, is
a normal and healthy result, not evidence the implementation is leaving
obvious wins on the table.

## 5. Correctness: the gradient discrepancy number

The benchmark printed `max gradient discrepancy: 2.50111e-12`, not exactly
`0`. That's expected and is not a bug. Floating-point addition is not
strictly associative, summing the same set of numbers in a different order
produces a result that differs in the last few bits of precision. The
single-threaded version sums all 4096 rows' gradients in one fixed order; the
multi-threaded version sums 12 partial sums in whatever order the threads
happen to finish and acquire the lock, then adds those together. Same numbers,
different order, a discrepancy at the level of `1e-12` is exactly what you'd
expect from accumulated floating-point rounding noise on a sum over millions
of double-precision operations, several orders of magnitude below anything
that would affect training. This is the correctness story to give if asked:
the two methods compute the same mathematical sum, the tiny numerical
difference is a property of floating-point arithmetic, not of the threading
logic being wrong.

## 6. What this does and doesn't demonstrate

**Does demonstrate:** correct use of `std::thread` for data-parallel batch
processing, `std::mutex` for synchronizing a shared accumulator, and a
measured, reproducible speedup with a sound explanation for why it isn't
perfectly linear.

**Doesn't demonstrate:** distributed systems. Everything here runs in one
process on one machine with shared memory; there's no network boundary, no
message passing, no partial-failure handling, none of the problems that
define "distributed" as opposed to "concurrent." If asked the difference in
an interview, that's the honest line: this is thread-level concurrency on
shared memory, not a distributed system, and the two terms shouldn't be used
interchangeably even though they often get lumped together casually.

## 7. One-line takeaway

Same model, same data, same result, **5.47x faster wall-clock training time**
on a 12-thread machine, achieved by splitting each batch's gradient
computation across threads and synchronizing only the final accumulation step
with a mutex, verified correct by checking that the parallel and serial
gradient sums match to within floating-point rounding noise.
