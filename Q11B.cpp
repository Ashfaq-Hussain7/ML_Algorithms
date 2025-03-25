#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>

// Function to generate a random integer array
std::vector<int> generateArray(int n, int minVal = 1, int maxVal = 1000) {
    std::vector<int> arr(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(minVal, maxVal);

    for (int &num : arr) {
        num = dist(gen);
    }
    return arr;
}

// Function for multithreaded sum
void threadSum(const std::vector<int> &arr, int start, int end, long long &sum) {
    sum = 0;
    for (int i = start; i < end; ++i) {
        sum += arr[i];
    }
}

// Function for multithreaded search
void threadSearch(const std::vector<int> &arr, int start, int end, int key, bool &found) {
    for (int i = start; i < end; ++i) {
        if (arr[i] == key) {
            found = true;
            return;
        }
    }
}

int main() {
    int n = 1000000;  // Array size
    int key = 500;    // Element to search
    int num_threads = 4;  // Number of threads

    std::vector<int> arr = generateArray(n);
    std::vector<std::thread> threads;
    std::vector<long long> partial_sums(num_threads, 0);
    std::vector<bool> partial_found(num_threads, false);
    int chunk_size = n / num_threads;

    // Multi-threaded Sum
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? n : start + chunk_size;
        threads.emplace_back(threadSum, std::cref(arr), start, end, std::ref(partial_sums[i]));
    }
    for (auto &t : threads) t.join();
    threads.clear();

    long long total_sum = 0;
    for (long long s : partial_sums) total_sum += s;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Multithreaded Sum: " << total_sum << "\n";
    std::cout << "Time Taken (Multithreaded Sum): " 
              << std::chrono::duration<double>(end_time - start_time).count() << " sec\n";

    // Multi-threaded Search
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? n : start + chunk_size;
        threads.emplace_back(threadSearch, std::cref(arr), start, end, key, std::ref(partial_found[i]));
    }
    for (auto &t : threads) t.join();

    bool found = false;
    for (bool f : partial_found) {
        if (f) {
            found = true;
            break;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Multithreaded Search: " << (found ? "Found" : "Not Found") << "\n";
    std::cout << "Time Taken (Multithreaded Search): " 
              << std::chrono::duration<double>(end_time - start_time).count() << " sec\n";

    return 0;
}
