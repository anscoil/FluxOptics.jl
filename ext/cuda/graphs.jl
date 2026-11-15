# Round trips replayed from a CUDA graph (see OpticalComponents.recorded_roundtrip). The
# first call records the round trip and launches the graph, every later call launches it
# on the stream of the task. The kernels are already compiled at that point: fp_solve!
# runs the same round trip with the sources just before. The graph keeps the buffers and
# the kernel arguments of the recording: valid for an operator, whose round trip always
# runs on the same buffers. A recording that failed is retried at the next call, the
# round trip running directly meanwhile.
mutable struct RecordedRoundtrip{F}
    roundtrip!::F
    exec::Union{Nothing, CuGraphExec}
end

function (r::RecordedRoundtrip)()
    isnothing(r.exec) && (r.exec = record(r.roundtrip!))
    isnothing(r.exec) ? r.roundtrip!() : CUDA.launch(r.exec)
    nothing
end

# capture saves the state of the garbage collector, turns it off, and restores the saved
# state at the end; that state is per thread. Hence one recording at a time, in a task
# stuck to the current thread: a recording task that waits on a lock (every kernel launch
# takes cufunction_lock, shared by all tasks) could otherwise resume on another thread,
# and two recordings overlapping on the same thread (the first ending before the second)
# would restore the off state of the first: in both cases the collector stays off for
# good. The task is stuck to the thread with jl_set_task_tid, an internal of Julia: the
# call with which Base itself sticks @async tasks to their thread (enq_work in task.jl),
# without sticking the calling task as @async does.
const RECORDING = ReentrantLock()

function record(roundtrip!)
    lock(RECORDING) do
        dev = device()
        task = Task(() -> record_on_new_stream(roundtrip!, dev))
        task.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, Threads.threadid() - 1)
        fetch(schedule(task))
    end
end

# Recording on a new stream:
# - finalizers already pending still run in other tasks during the capture, and freeing
#   an array of the stream of the task while that stream records fails (cuMemFreeAsync,
#   invalid value); the new stream only owns the buffers of the round trip, alive while
#   it records;
# - non-blocking: work on the legacy stream in another task (synchronous copies of cuFFT
#   when it builds a plan, for instance) would depend on a blocking recording stream,
#   which invalidates the recording;
# - relaxed mode: other tasks keep using the device during the recording, even one that
#   runs on the recording thread while the recording task waits on a lock.
# Any other invalidation returns nothing (a synchronization of the recording stream in
# the round trip, or of the whole device in another task), with a warning.
function record_on_new_stream(roundtrip!, dev)
    device!(dev)
    graph = stream!(CuStream(; flags = CUDA.STREAM_NON_BLOCKING)) do
        CUDA.capture(roundtrip!; flags = CUDA.STREAM_CAPTURE_MODE_RELAXED,
                     throw_error = false)
    end
    if isnothing(graph)
        @warn "CUDA graph recording invalidated: round trip run directly, recording " *
              "retried at the next product" maxlog = 1
        return nothing
    end
    CUDA.instantiate(graph)
end

function OpticalComponents.recorded_roundtrip(roundtrip!, ::CUDABackend)
    RecordedRoundtrip(roundtrip!, nothing)
end
