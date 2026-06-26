

#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <stdexcept>

struct Matrix {
    int rows, cols;
    std::vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c, double init = 0.0) : rows(r), cols(c), data(r * c, init) {}

    double& at(int r, int c) { return data[r * cols + c]; }
    double at(int r, int c) const { return data[r * cols + c]; }

    static Matrix randn(int r, int c, std::mt19937& gen, double stddev) {
        std::normal_distribution<double> dist(0.0, stddev);
        Matrix m(r, c);
        for (auto& v : m.data) v = dist(gen);
        return m;
    }

    Matrix transpose() const {
        Matrix out(cols, rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                out.at(j, i) = at(i, j);
        return out;
    }

    void add_in_place(const Matrix& other) {
        assert(rows == other.rows && cols == other.cols);
        for (size_t i = 0; i < data.size(); ++i) data[i] += other.data[i];
    }

    void scale_in_place(double s) {
        for (auto& v : data) v *= s;
    }
};

Matrix matmul(const Matrix& a, const Matrix& b) {
    assert(a.cols == b.rows);
    Matrix out(a.rows, b.cols);
    for (int i = 0; i < a.rows; ++i) {
        for (int k = 0; k < a.cols; ++k) {
            double aik = a.at(i, k);
            if (aik == 0.0) continue;
            for (int j = 0; j < b.cols; ++j) {
                out.at(i, j) += aik * b.at(k, j);
            }
        }
    }
    return out;
}

void add_row_broadcast(Matrix& m, const Matrix& bias) {
    assert(bias.rows == 1 && bias.cols == m.cols);
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            m.at(i, j) += bias.at(0, j);
}

Matrix relu(const Matrix& z) {
    Matrix out = z;
    for (auto& v : out.data) v = std::max(0.0, v);
    return out;
}

Matrix relu_backward(const Matrix& grad_out, const Matrix& z) {
    Matrix out = grad_out;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (z.data[i] <= 0.0) out.data[i] = 0.0;
    return out;
}

Matrix softmax(const Matrix& z) {
    Matrix out(z.rows, z.cols);
    for (int i = 0; i < z.rows; ++i) {
        double maxv = -1e300;
        for (int j = 0; j < z.cols; ++j) maxv = std::max(maxv, z.at(i, j));
        double sum = 0.0;
        for (int j = 0; j < z.cols; ++j) {
            double e = std::exp(z.at(i, j) - maxv);
            out.at(i, j) = e;
            sum += e;
        }
        for (int j = 0; j < z.cols; ++j) out.at(i, j) /= sum;
    }
    return out;
}

struct Network {
    Matrix W1, b1, W2, b2;

    Network() = default;
    Network(int input_dim, int hidden_dim, int output_dim, std::mt19937& gen) {
        double s1 = std::sqrt(2.0 / input_dim);
        double s2 = std::sqrt(2.0 / hidden_dim);
        W1 = Matrix::randn(input_dim, hidden_dim, gen, s1);
        b1 = Matrix(1, hidden_dim, 0.0);
        W2 = Matrix::randn(hidden_dim, output_dim, gen, s2);
        b2 = Matrix(1, output_dim, 0.0);
    }
};

struct Gradients {
    Matrix dW1, db1, dW2, db2;

    Gradients() = default;
    Gradients(int input_dim, int hidden_dim, int output_dim) {
        dW1 = Matrix(input_dim, hidden_dim, 0.0);
        db1 = Matrix(1, hidden_dim, 0.0);
        dW2 = Matrix(hidden_dim, output_dim, 0.0);
        db2 = Matrix(1, output_dim, 0.0);
    }

    void add_in_place(const Gradients& other) {
        dW1.add_in_place(other.dW1);
        db1.add_in_place(other.db1);
        dW2.add_in_place(other.dW2);
        db2.add_in_place(other.db2);
    }

    void scale_in_place(double s) {
        dW1.scale_in_place(s);
        db1.scale_in_place(s);
        dW2.scale_in_place(s);
        db2.scale_in_place(s);
    }

    double max_abs_diff(const Gradients& other) const {
        double m = 0.0;
        auto cmp = [&](const std::vector<double>& a, const std::vector<double>& b) {
            for (size_t i = 0; i < a.size(); ++i)
                m = std::max(m, std::abs(a[i] - b[i]));
        };
        cmp(dW1.data, other.dW1.data);
        cmp(db1.data, other.db1.data);
        cmp(dW2.data, other.dW2.data);
        cmp(db2.data, other.db2.data);
        return m;
    }
};

Gradients compute_gradients_chunk(const Network& net,
                                   const Matrix& X, const Matrix& Y,
                                   int row_start, int row_end,
                                   double& out_loss) {
    int n = row_end - row_start;
    int input_dim = net.W1.rows;
    int hidden_dim = net.W1.cols;
    int output_dim = net.W2.cols;

    Matrix Xc(n, input_dim);
    Matrix Yc(n, output_dim);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < input_dim; ++j) Xc.at(i, j) = X.at(row_start + i, j);
        for (int j = 0; j < output_dim; ++j) Yc.at(i, j) = Y.at(row_start + i, j);
    }

    Matrix Z1 = matmul(Xc, net.W1);
    add_row_broadcast(Z1, net.b1);
    Matrix A1 = relu(Z1);

    Matrix Z2 = matmul(A1, net.W2);
    add_row_broadcast(Z2, net.b2);
    Matrix A2 = softmax(Z2);

    double loss = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < output_dim; ++j)
            if (Yc.at(i, j) > 0.0)
                loss -= Yc.at(i, j) * std::log(std::max(A2.at(i, j), 1e-12));
    out_loss = loss;

    Matrix dZ2(n, output_dim);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < output_dim; ++j)
            dZ2.at(i, j) = A2.at(i, j) - Yc.at(i, j);

    Gradients g(input_dim, hidden_dim, output_dim);

    g.dW2 = matmul(A1.transpose(), dZ2);
    for (int j = 0; j < output_dim; ++j) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += dZ2.at(i, j);
        g.db2.at(0, j) = s;
    }

    Matrix dA1 = matmul(dZ2, net.W2.transpose());
    Matrix dZ1 = relu_backward(dA1, Z1);

    g.dW1 = matmul(Xc.transpose(), dZ1);
    for (int j = 0; j < hidden_dim; ++j) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += dZ1.at(i, j);
        g.db1.at(0, j) = s;
    }

    return g;
}

Gradients compute_gradients_single_threaded(const Network& net,
                                              const Matrix& X, const Matrix& Y,
                                              double& out_loss) {
    return compute_gradients_chunk(net, X, Y, 0, X.rows, out_loss);
}

Gradients compute_gradients_multi_threaded(const Network& net,
                                             const Matrix& X, const Matrix& Y,
                                             int num_threads,
                                             double& out_loss) {
    int input_dim = net.W1.rows;
    int hidden_dim = net.W1.cols;
    int output_dim = net.W2.cols;
    int n = X.rows;

    Gradients shared_grad(input_dim, hidden_dim, output_dim);
    double shared_loss = 0.0;
    std::mutex grad_mutex;

    std::vector<std::thread> workers;
    int chunk = (n + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int row_start = t * chunk;
        int row_end = std::min(n, row_start + chunk);
        if (row_start >= row_end) continue;

        workers.emplace_back([&, row_start, row_end]() {
            double local_loss = 0.0;
            Gradients local_grad = compute_gradients_chunk(net, X, Y, row_start, row_end, local_loss);

            std::lock_guard<std::mutex> lock(grad_mutex);
            shared_grad.add_in_place(local_grad);
            shared_loss += local_loss;
        });
    }

    for (auto& th : workers) th.join();

    out_loss = shared_loss;
    return shared_grad;
}

void apply_update(Network& net, const Gradients& g, double lr) {
    for (size_t i = 0; i < net.W1.data.size(); ++i) net.W1.data[i] -= lr * g.dW1.data[i];
    for (size_t i = 0; i < net.b1.data.size(); ++i) net.b1.data[i] -= lr * g.db1.data[i];
    for (size_t i = 0; i < net.W2.data.size(); ++i) net.W2.data[i] -= lr * g.dW2.data[i];
    for (size_t i = 0; i < net.b2.data.size(); ++i) net.b2.data[i] -= lr * g.db2.data[i];
}

double forward_accuracy(Network& net, const Matrix& X, const Matrix& Y) {
    Matrix Z1 = matmul(X, net.W1); add_row_broadcast(Z1, net.b1); Matrix A1 = relu(Z1);
    Matrix Z2 = matmul(A1, net.W2); add_row_broadcast(Z2, net.b2); Matrix A2 = softmax(Z2);

    int correct = 0;
    for (int i = 0; i < X.rows; ++i) {
        int pred = 0; double best = -1.0;
        for (int j = 0; j < A2.cols; ++j) if (A2.at(i, j) > best) { best = A2.at(i, j); pred = j; }
        int actual = 0; double best_y = -1.0;
        for (int j = 0; j < Y.cols; ++j) if (Y.at(i, j) > best_y) { best_y = Y.at(i, j); actual = j; }
        if (pred == actual) ++correct;
    }
    return static_cast<double>(correct) / X.rows;
}

struct Dataset { Matrix X, Y; };

Dataset load_covtype_csv(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open file: " + path);

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (content.empty()) throw std::runtime_error("File is empty: " + path);

    size_t pos = content.find('\n');
    if (pos == std::string::npos) throw std::runtime_error("File has no data rows: " + path);
    pos += 1; // skip header line

    std::vector<std::vector<double>> rows;
    rows.reserve(600000);
    std::vector<int> labels;
    labels.reserve(600000);

    const char* base = content.c_str();
    size_t total_len = content.size();

    while (pos < total_len) {
        size_t line_end = content.find('\n', pos);
        if (line_end == std::string::npos) line_end = total_len;
        if (line_end == pos) { pos = line_end + 1; continue; } // blank line

        std::vector<double> vals;
        vals.reserve(55);
        const char* p = base + pos;
        const char* line_end_ptr = base + line_end;
        while (p < line_end_ptr) {
            char* next = nullptr;
            double v = std::strtod(p, &next);
            vals.push_back(v);
            if (next == p) break; // malformed field, bail on this line's remainder
            p = next;
            if (p < line_end_ptr && *p == ',') ++p;
        }

        if (vals.size() >= 2) {
            int label = static_cast<int>(vals.back());
            vals.pop_back();
            rows.push_back(std::move(vals));
            labels.push_back(label - 1); // zero-index: 1..7 -> 0..6
        }

        pos = line_end + 1;
    }

    int n = static_cast<int>(rows.size());
    if (n == 0) throw std::runtime_error("No data rows parsed from: " + path);
    int input_dim = static_cast<int>(rows[0].size());
    int output_dim = 7;

    Matrix X(n, input_dim);
    Matrix Y(n, output_dim, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < input_dim; ++j) X.at(i, j) = rows[i][j];
        Y.at(i, labels[i]) = 1.0;
    }
    return {X, Y};
}

void normalize_continuous_columns(Matrix& X, int num_continuous) {
    int n = X.rows;
    for (int j = 0; j < num_continuous; ++j) {
        double mean = 0.0;
        for (int i = 0; i < n; ++i) mean += X.at(i, j);
        mean /= n;
        double var = 0.0;
        for (int i = 0; i < n; ++i) { double d = X.at(i, j) - mean; var += d * d; }
        double stddev = std::sqrt(var / n) + 1e-8;
        for (int i = 0; i < n; ++i) X.at(i, j) = (X.at(i, j) - mean) / stddev;
    }
}

void shuffle_dataset(Matrix& X, Matrix& Y, std::mt19937& gen) {
    int n = X.rows;
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), gen);

    Matrix Xs(X.rows, X.cols);
    Matrix Ys(Y.rows, Y.cols);
    for (int i = 0; i < n; ++i) {
        int src = perm[i];
        for (int j = 0; j < X.cols; ++j) Xs.at(i, j) = X.at(src, j);
        for (int j = 0; j < Y.cols; ++j) Ys.at(i, j) = Y.at(src, j);
    }
    X = std::move(Xs);
    Y = std::move(Ys);
}

void slice_rows(const Matrix& X, const Matrix& Y, int start, int end, Matrix& Xo, Matrix& Yo) {
    int n = end - start;
    Xo = Matrix(n, X.cols);
    Yo = Matrix(n, Y.cols);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < X.cols; ++j) Xo.at(i, j) = X.at(start + i, j);
        for (int j = 0; j < Y.cols; ++j) Yo.at(i, j) = Y.at(start + i, j);
    }
}


int main(int argc, char** argv) {
    std::string csv_path = argc > 1 ? argv[1] : "covtype.csv";

    const int hidden_dim = 128;
    const int batch_size = 4096;
    const int num_epochs = 5;
    const double lr = 5e-3;
    const int num_continuous_features = 10;

    unsigned hw_threads = std::thread::hardware_concurrency();
    int num_threads = hw_threads == 0 ? 4 : static_cast<int>(hw_threads);

    std::cout << "Loading " << csv_path << " ...\n";
    auto load_start = std::chrono::high_resolution_clock::now();
    Dataset full = load_covtype_csv(csv_path);
    auto load_end = std::chrono::high_resolution_clock::now();
    std::cout << "Loaded " << full.X.rows << " rows, " << full.X.cols
              << " features in "
              << std::chrono::duration<double>(load_end - load_start).count() << " s\n";

    int input_dim = full.X.cols;
    int output_dim = full.Y.cols;

    normalize_continuous_columns(full.X, num_continuous_features);

    std::mt19937 gen(42);
    shuffle_dataset(full.X, full.Y, gen);

    int n_total = full.X.rows;
    int n_test = n_total / 10;
    int n_train = n_total - n_test;

    Matrix X_train, Y_train, X_test, Y_test;
    slice_rows(full.X, full.Y, 0, n_train, X_train, Y_train);
    slice_rows(full.X, full.Y, n_train, n_total, X_test, Y_test);

    std::cout << "Train: " << X_train.rows << " rows, Test: " << X_test.rows << " rows\n";
    std::cout << "Architecture: " << input_dim << " -> " << hidden_dim
              << " (ReLU) -> " << output_dim << " (softmax)\n";
    std::cout << "Threads available: " << num_threads << "\n\n";

    std::mt19937 init_gen(123);
    Network net(input_dim, hidden_dim, output_dim, init_gen);

    {
        Matrix Xb, Yb;
        slice_rows(X_train, Y_train, 0, std::min(batch_size, X_train.rows), Xb, Yb);

        Network net_copy = net; // identical weights for a fair comparison

        double loss_s = 0.0, loss_m = 0.0;
        auto t0 = std::chrono::high_resolution_clock::now();
        Gradients g_single = compute_gradients_single_threaded(net, Xb, Yb, loss_s);
        auto t1 = std::chrono::high_resolution_clock::now();
        Gradients g_multi = compute_gradients_multi_threaded(net_copy, Xb, Yb, num_threads, loss_m);
        auto t2 = std::chrono::high_resolution_clock::now();

        double single_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double multi_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        std::cout << "[Benchmark on first batch of " << Xb.rows << " rows]\n";
        std::cout << "  single-threaded: " << single_ms << " ms\n";
        std::cout << "  multi-threaded (" << num_threads << " threads): " << multi_ms << " ms\n";
        std::cout << "  speedup: " << (single_ms / multi_ms) << "x\n";
        std::cout << "  max gradient discrepancy (correctness check): "
                  << g_single.max_abs_diff(g_multi) << " (should be ~0)\n\n";
    }

    // ---- Actual training loop, using the multi-threaded path throughout ----
    std::vector<int> train_idx(X_train.rows);
    std::iota(train_idx.begin(), train_idx.end(), 0);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::shuffle(train_idx.begin(), train_idx.end(), gen);

        // Reorder X_train/Y_train according to the shuffled indices for
        // this epoch (kept simple: rebuild shuffled copies once per epoch).
        Matrix Xe(X_train.rows, X_train.cols);
        Matrix Ye(Y_train.rows, Y_train.cols);
        for (int i = 0; i < X_train.rows; ++i) {
            int src = train_idx[i];
            for (int j = 0; j < X_train.cols; ++j) Xe.at(i, j) = X_train.at(src, j);
            for (int j = 0; j < Y_train.cols; ++j) Ye.at(i, j) = Y_train.at(src, j);
        }

        double epoch_loss = 0.0;
        int num_batches = 0;
        auto epoch_start = std::chrono::high_resolution_clock::now();

        for (int start = 0; start < Xe.rows; start += batch_size) {
            int end = std::min(Xe.rows, start + batch_size);
            Matrix Xb, Yb;
            slice_rows(Xe, Ye, start, end, Xb, Yb);

            double batch_loss = 0.0;
            Gradients g = compute_gradients_multi_threaded(net, Xb, Yb, num_threads, batch_loss);
            g.scale_in_place(1.0 / Xb.rows);
            apply_update(net, g, lr);

            epoch_loss += batch_loss;
            ++num_batches;
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double train_acc = forward_accuracy(net, X_train, Y_train);
        double test_acc = forward_accuracy(net, X_test, Y_test);

        std::cout << "Epoch " << epoch
                  << " | avg loss: " << (epoch_loss / Xe.rows)
                  << " | train acc: " << train_acc
                  << " | test acc: " << test_acc
                  << " | time: " << std::chrono::duration<double>(epoch_end - epoch_start).count() << " s\n";
    }

    return 0;
}