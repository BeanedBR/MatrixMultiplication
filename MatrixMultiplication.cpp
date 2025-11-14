#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <mutex>

using Matrix = std::vector<double>;

// Helper to index into the 1D vector as if it were 2D (n x n)
inline double& at(Matrix& M, std::size_t n, std::size_t row, std::size_t col) {
    return M[row * n + col];
}

inline const double& at(const Matrix& M, std::size_t n, std::size_t row, std::size_t col) {
    return M[row * n + col];
}

// Generate an n x n matrix with random values in [0, 1)
Matrix random_matrix(std::size_t n) {
    Matrix M(n * n);
    std::mt19937 rng(42); // fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (std::size_t i = 0; i < n * n; ++i) {
        M[i] = dist(rng);
    }
    return M;
}

// Sequential matrix multiplication: C = A * B
void matmul_sequential(const Matrix& A, const Matrix& B, Matrix& C, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += at(A, n, i, k) * at(B, n, k, j);
            }
            at(C, n, i, j) = sum;
        }
    }
}

// Worker function for parallel multiplication: computes rows [row_start, row_end)
void matmul_worker(const Matrix& A, const Matrix& B, Matrix& C,
    std::size_t n,
    std::size_t row_start, std::size_t row_end,
    unsigned int thread_id,
    bool show_progress,
    std::mutex& print_mutex) {
    if (show_progress) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << "[Thread " << thread_id << "] Starting rows "
            << row_start << " to " << (row_end == 0 ? 0 : row_end - 1) << "\n";
    }

    for (std::size_t i = row_start; i < row_end; ++i) {
        if (show_progress) {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cout << "[Thread " << thread_id << "] Computing row " << i << "\n";
        }

        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += at(A, n, i, k) * at(B, n, k, j);
            }
            at(C, n, i, j) = sum;
        }
    }

    if (show_progress) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << "[Thread " << thread_id << "] Finished.\n";
    }
}

// Parallel matrix multiplication using std::thread
void matmul_parallel(const Matrix& A, const Matrix& B, Matrix& C,
    std::size_t n,
    unsigned int num_threads,
    bool show_progress) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) {
            num_threads = 2; // fallback
        }
    }

    // Optional: avoid spawning more threads than rows
    if (num_threads > n) {
        num_threads = static_cast<unsigned int>(n);
    }

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::size_t rows_per_thread = n / num_threads;
    std::size_t extra_rows = n % num_threads;

    std::size_t current_row = 0;

    std::mutex print_mutex; // protects std::cout when show_progress is true

    for (unsigned int t = 0; t < num_threads; ++t) {
        std::size_t start_row = current_row;
        std::size_t rows_for_this_thread = rows_per_thread + (t < extra_rows ? 1 : 0);
        std::size_t end_row = start_row + rows_for_this_thread;

        threads.emplace_back(
            matmul_worker,
            std::cref(A), std::cref(B), std::ref(C),
            n,
            start_row, end_row,
            t,                      // thread_id
            show_progress,
            std::ref(print_mutex)
        );

        current_row = end_row;
    }

    for (auto& th : threads) {
        th.join();
    }
}

// Compute maximum absolute difference between two matrices (for correctness check)
double max_difference(const Matrix& X, const Matrix& Y) {
    if (X.size() != Y.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double max_diff = 0.0;
    for (std::size_t i = 0; i < X.size(); ++i) {
        double diff = std::abs(X[i] - Y[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

int main() {
    std::size_t n;
    unsigned int num_threads;

    std::cout << "Parallel Matrix Multiplication Demo\n";
    std::cout << "-----------------------------------\n";
    std::cout << "Enter matrix size n (e.g. 300, 500, 700): ";
    if (!(std::cin >> n) || n == 0) {
        std::cerr << "Invalid matrix size.\n";
        return 1;
    }

    std::cout << "Enter number of threads (0 = use hardware_concurrency): ";
    if (!(std::cin >> num_threads)) {
        std::cerr << "Invalid number of threads.\n";
        return 1;
    }

    char show_choice;
    bool show_progress = false;
    std::cout << "Show thread processes? (y/n): ";
    if (!(std::cin >> show_choice)) {
        std::cerr << "Invalid choice.\n";
        return 1;
    }
    if (show_choice == 'y' || show_choice == 'Y') {
        show_progress = true;
    }

    std::cout << "\nGenerating random matrices A and B (" << n << " x " << n << ")...\n";
    Matrix A = random_matrix(n);
    Matrix B = random_matrix(n);

    Matrix C_seq(n * n, 0.0);
    Matrix C_par(n * n, 0.0);

    using clock = std::chrono::high_resolution_clock;

    // Sequential
    std::cout << "Running sequential multiplication...\n";
    auto start_seq = clock::now();
    matmul_sequential(A, B, C_seq, n);
    auto end_seq = clock::now();
    double seq_ms = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();

    // Parallel
    std::cout << "Running parallel multiplication with " << num_threads << " thread(s)...\n";
    if (show_progress) {
        std::cout << "NOTE: Showing thread processes will significantly slow down the run.\n";
        std::cout << "      Use this mode for understanding/demo, not for performance measurements.\n";
    }

    auto start_par = clock::now();
    matmul_parallel(A, B, C_par, n, num_threads, show_progress);
    auto end_par = clock::now();
    double par_ms = std::chrono::duration<double, std::milli>(end_par - start_par).count();

    // Correctness check
    double max_diff = max_difference(C_seq, C_par);

    // Speedup & efficiency
    double speedup = seq_ms / par_ms;
    double efficiency = speedup / static_cast<double>(num_threads == 0 ? 1 : num_threads);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nResults:\n";
    std::cout << "Sequential time: " << seq_ms << " ms\n";
    std::cout << "Parallel   time: " << par_ms << " ms\n";
    std::cout << "Speedup        : " << speedup << "x\n";
    std::cout << "Efficiency     : " << efficiency * 100.0 << " %\n";
    std::cout << "Max difference between results: " << max_diff << "\n";

    return 0;
}
