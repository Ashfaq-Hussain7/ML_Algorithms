#include <iostream>
#include <vector>
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

// Function to find the sum of elements in an array
long long sequentialSum(const std::vector<int> &arr) {
    long long sum = 0;
    for (int num : arr) {
        sum += num;
    }
    return sum;
}

// Function to search for a key element in an array
bool sequentialSearch(const std::vector<int> &arr, int key) {
    for (int num : arr) {
        if (num == key) return true;
    }
    return false;
}

int main() {
    int n = 1000000;  // Array size
    int key = 500;    // Element to search

    std::vector<int> arr = generateArray(n);

    // Measure execution time for sum
    auto start_time = std::chrono::high_resolution_clock::now();
    long long sum = sequentialSum(arr);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Sequential Sum: " << sum << "\n";
    std::cout << "Time Taken (Sequential Sum): " 
              << std::chrono::duration<double>(end_time - start_time).count() << " sec\n";

    // Measure execution time for search
    start_time = std::chrono::high_resolution_clock::now();
    bool found = sequentialSearch(arr, key);
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Sequential Search: " << (found ? "Found" : "Not Found") << "\n";
    std::cout << "Time Taken (Sequential Search): " 
              << std::chrono::duration<double>(end_time - start_time).count() << " sec\n";

    return 0;
}
