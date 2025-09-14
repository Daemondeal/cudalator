#include "Runtime.hpp"
#include "Stats.hpp"
#include "cuda_compat.hpp"
#include "UserProvided.hpp"
#include <vector>

Circuit::Circuit(int number_of_circuits)
    : m_num_circuits(number_of_circuits), m_cycles(0), m_stats({}) {
    // allocate on the gpu
    size_t states_size = sizeof(StateType) * m_num_circuits;
    size_t diffs_size = sizeof(DiffType) * m_num_circuits;
    CUDA_CHECK(cudaMalloc(&d_states, states_size));
    CUDA_CHECK(cudaMalloc(&d_previous_states, states_size));
    CUDA_CHECK(cudaMalloc(&d_diffs, diffs_size));

    // apparently a locked page allows for faster transfers
    CUDA_CHECK(cudaHostAlloc(&h_diff_sum, sizeof(DiffType), cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&d_diff_sum, sizeof(DiffType)));
    CUDA_CHECK(cudaMemset(d_states, 0, states_size));
    CUDA_CHECK(cudaMemset(d_previous_states, 0, states_size));

    m_stats.number_of_circuits = m_num_circuits;
    m_stats.state_array_size = states_size * 2;
    m_stats.diff_array_size = diffs_size;

    m_processes = make_processes();
    first_eval();
}

Circuit::~Circuit() {
    cudaFree(d_states);
    cudaFree(d_previous_states);
    cudaFree(d_diffs);
    cudaFreeHost(h_diff_sum);
}

// We copy data to the GPU, we modify it and then copy it back
void Circuit::apply_input() {
    const int threads_per_block = 256;
    const int blocks =
        (m_num_circuits + threads_per_block - 1) / threads_per_block;

    cudalator_apply_input<<<blocks, threads_per_block>>>(d_states, m_cycles, m_num_circuits);
}


// State diff reduction kernel
__global__ void reduction_change(DiffType *diffs, DiffType *res, size_t x_start, size_t y_start, size_t limit) {
    int tid = threadIdx.x;

    int block_id_x = blockIdx.x * blockDim.x + x_start;

    int global_id_x = blockIdx.x * blockDim.x + tid + x_start;
    int global_id_y = blockIdx.y + y_start;

    int shared_limit = limit - block_id_x;

    extern __shared__ ChangeType shared[];
    if (global_id_x < limit) {
        shared[tid] = diffs[global_id_x].change[global_id_y];
    }
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset && tid < shared_limit && (tid+offset) < shared_limit) {
            shared[tid] = change_combine(shared[tid], shared[tid + offset]);
        }
        __syncthreads();
    }

    // Atomically apply change
    if (tid == 0) {
        int assumed, old_change;
        int *dest = (int*)&res->change[global_id_y];

        old_change = *dest;
        do {
            assumed = old_change;
            ChangeType combined = change_combine((ChangeType)assumed, shared[0]);
            old_change = atomicCAS(dest, assumed, (int)combined);
        } while (assumed != old_change);
    }
}

void Circuit::eval() {
    std::vector<ProcType> ready_queue;
    const int threads_per_block = 256;
    const int blocks =
        (m_num_circuits + threads_per_block - 1) / threads_per_block;

    m_stats.start_counter(PerfEvent::DoDeltaCycle);
    while (true) {
        m_stats.start_counter(PerfEvent::DoIteration);

        m_stats.start_counter(PerfEvent::CalculateStateDiff);
        // launching the diff kernel
        state_calculate_diff<<<blocks, threads_per_block>>>(
            d_previous_states, d_states, d_diffs, m_num_circuits);
        CUDA_CHECK(cudaGetLastError());
        m_stats.stop_counter(PerfEvent::CalculateStateDiff);


        m_stats.start_counter(PerfEvent::MergeStateDiff);
        CUDA_CHECK(cudaMemset(d_diff_sum, 0, sizeof(DiffType)));

        constexpr uint32_t MAX_BLOCK_SIZE = 256;
        constexpr uint32_t MAX_GRID_SIZE_X = 32;
        constexpr uint32_t MAX_GRID_SIZE_Y = 32;

        uint32_t y_remaining = sizeof(d_diff_sum->change)/sizeof(ChangeType);
        uint32_t y_start = 0;
        while (y_remaining > 0) {
            uint32_t y_done = std::min(y_remaining, MAX_GRID_SIZE_Y);

            uint32_t x_remaining = m_num_circuits;
            uint32_t x_start = 0;
            while (x_remaining > 0) {
                uint32_t x_done = std::min(x_remaining, MAX_GRID_SIZE_X * MAX_BLOCK_SIZE);

                dim3 block_size(MAX_BLOCK_SIZE, 1);
                dim3 grid_size((x_done + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE, y_done);
                size_t shared_memory_size = MAX_BLOCK_SIZE * sizeof(ChangeType);
                reduction_change<<<grid_size, block_size, shared_memory_size>>>(
                    d_diffs,
                    d_diff_sum,
                    x_start,
                    y_start,
                    x_start+x_done
                );

                CUDA_CHECK( cudaPeekAtLastError() );

                x_remaining -= x_done;
                x_start += x_done;
            }
            y_remaining -= y_done;
            y_start += y_done;
        }
        m_stats.stop_counter(PerfEvent::MergeStateDiff);


        m_stats.start_counter(PerfEvent::MemcopyStateForPopulating);
        // coying the diff results
        CUDA_CHECK(cudaMemcpy(h_diff_sum, d_diff_sum,
                              sizeof(DiffType),
                              cudaMemcpyDeviceToHost));
        m_stats.stop_counter(PerfEvent::MemcopyStateForPopulating);

        m_stats.start_counter(PerfEvent::PopulateReadyQueue);
        // cpu computation of the ready queue
        for (const auto& proc : m_processes) {
            bool should_run = false;
            for (auto [signal_idx, change_type] : proc.sensitivity) {
                auto actual = h_diff_sum->change[signal_idx];
                if ((change_type == ChangeType::Change &&
                     actual != ChangeType::NoChange) ||
                    (change_type == actual)) {
                    should_run = true;
                    break;
                }
            }
            if (should_run)
                ready_queue.push_back(proc);
        }
        m_stats.stop_counter(PerfEvent::PopulateReadyQueue);

        // if empy, it's already stable
        if (ready_queue.empty()) {
            m_stats.stop_counter(PerfEvent::DoIteration);
            break;
        }
        m_stats.kernels_launched += ready_queue.size();

        m_stats.start_counter(PerfEvent::CloneStates);
        // save current state for next diff computation
        CUDA_CHECK(cudaMemcpy(d_previous_states, d_states,
                              sizeof(StateType) * m_num_circuits,
                              cudaMemcpyDeviceToDevice));
        m_stats.stop_counter(PerfEvent::CloneStates);

        m_stats.start_counter(PerfEvent::RunKernels);
        // launch of all the ready kernels
        for (const auto& proc : ready_queue) {
            run_process<<<blocks, threads_per_block>>>(d_states, m_num_circuits,
                                                       proc.id);
            CUDA_CHECK(cudaGetLastError());
        }
        m_stats.stop_counter(PerfEvent::RunKernels);

        ready_queue.clear();
        m_stats.stop_counter(PerfEvent::DoIteration);
        m_stats.iterations_done++;
    }
    // end of delta cycle step

    m_cycles++;
    m_stats.delta_times_ran++;

    // state copy before next eval call
    CUDA_CHECK(cudaMemcpy(d_previous_states, d_states,
                          sizeof(StateType) * m_num_circuits,
                          cudaMemcpyDeviceToDevice));
    m_stats.stop_counter(PerfEvent::DoDeltaCycle);

    dump_to_vcd();
}

void Circuit::first_eval() {
    const int threads_per_block = 256;
    const int blocks =
        (m_num_circuits + threads_per_block - 1) / threads_per_block;
    for (const auto& proc : m_processes) {
        run_process<<<blocks, threads_per_block>>>(d_states, m_num_circuits,
                                                   proc.id);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    eval();
    m_cycles--;
    m_stats.kernels_launched += m_processes.size();
}

// no need to do anything below here

void Circuit::open_vcd(const std::string& path, int circuit_idx) {
    auto fp = std::make_unique<fmt::ostream>(fmt::output_file(path));
    m_vcd.emplace(std::move(fp), circuit_idx);
    m_vcd->print_header();
}

void Circuit::dump_to_vcd() {
    if (m_vcd.has_value()) {
        std::vector<StateType> h_states(m_num_circuits);
        CUDA_CHECK(cudaMemcpy(h_states.data(), d_states,
                              sizeof(StateType) * m_num_circuits,
                              cudaMemcpyDeviceToHost));
        m_vcd->dump(h_states, m_cycles);
    }
}
