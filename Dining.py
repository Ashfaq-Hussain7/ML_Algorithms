import threading
import time
import random

# Number of philosophers and forks
NUM_PHILOSOPHERS = 5
MAX_MEALS = 3  # Maximum meals per philosopher

# Lock for each fork
forks = [threading.Lock() for _ in range(NUM_PHILOSOPHERS)]
# Count meals for each philosopher
meals_eaten = [0] * NUM_PHILOSOPHERS

# Lock for updating shared state (meal counts)
meal_lock = threading.Lock()

def philosopher(philosopher_id):
    global meals_eaten
    left_fork = philosopher_id
    right_fork = (philosopher_id + 1) % NUM_PHILOSOPHERS

    while True:
        # Check if the philosopher has eaten the maximum number of meals
        with meal_lock:
            if meals_eaten[philosopher_id] >= MAX_MEALS:
                print(f"Philosopher {philosopher_id} has finished eating {MAX_MEALS} meals and is leaving.")
                break

        print(f"Philosopher {philosopher_id} is thinking.")
        time.sleep(random.uniform(1, 3))  # Simulate thinking

        print(f"Philosopher {philosopher_id} is hungry.")
        # To prevent deadlock, always pick the lower-numbered fork first
        first_fork = min(left_fork, right_fork)
        second_fork = max(left_fork, right_fork)

        # Try to acquire both forks
        with forks[first_fork]:
            with forks[second_fork]:
                print(f"Philosopher {philosopher_id} is eating.")
                time.sleep(random.uniform(1, 2))  # Simulate eating
                # Update meal count
                with meal_lock:
                    meals_eaten[philosopher_id] += 1

        print(f"Philosopher {philosopher_id} has finished eating.")

def main():
    # Create and start threads for each philosopher
    threads = [
        threading.Thread(target=philosopher, args=(i,))
        for i in range(NUM_PHILOSOPHERS)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()  # Wait for all threads to complete

    print("All philosophers have finished eating.")

if __name__ == "__main__":
    main()
