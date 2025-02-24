import threading
import time
import random

# Shared list for orders
orders = []
orders_lock = threading.Lock()  # Lock for thread-safe operations

# Function to simulate placing orders
def place_orders(order_count):
    global orders
    for i in range(order_count):
        order = f"Order-{i+1}"
        with orders_lock:
            orders.append(order)
            print(f"Placed: {order}")
        time.sleep(random.uniform(0.5, 1.5))  # Simulate random delay in placing orders

# Function to simulate processing orders
def process_orders():
    global orders
    while True:
        with orders_lock:
            if orders:
                order = orders.pop(0)
                print(f"Processed: {order}")
            else:
                break
        time.sleep(random.uniform(1, 2))  # Simulate random delay in processing orders

# Main function
def main():
    num_orders = 10
    place_thread = threading.Thread(target=place_orders, args=(num_orders,))
    process_thread = threading.Thread(target=process_orders)

    # Start threads
    place_thread.start()
    time.sleep(2)  # Ensure some orders are placed before starting to process
    process_thread.start()

    # Wait for threads to complete
    place_thread.join()
    process_thread.join()
    print("All orders placed and processed.")

if __name__ == "__main__":
    main()
