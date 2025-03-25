#include <iostream>
#include <thread>

// Function to print first n natural numbers
void printNaturalNumbers(int n) {
    for (int i = 1; i <= n; i++) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

int main() {
    int n;
    std::cout << "Enter the value of n: ";
    std::cin >> n;

    // Create a thread
    std::thread t(printNaturalNumbers, n);

    // Wait for thread to finish execution
    t.join();

    return 0;
}
