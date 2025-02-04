def m_way_merge_sort(arr, m):
    """Performs M-way Merge Sort on the given array."""
    if len(arr) <= 1:
        return arr

    # Step 1: Divide the array into M parts
    subarrays = []
    n = len(arr)
    part_size = (n + m - 1) // m  # Calculate size of each part, rounded up
    for i in range(0, n, part_size):
        subarrays.append(arr[i:i + part_size])

    # Step 2: Recursively sort each part
    sorted_subarrays = [m_way_merge_sort(subarray, m) for subarray in subarrays]

    # Step 3: Merge the M sorted parts
    return merge_m(sorted_subarrays)

def merge_m(sorted_subarrays):
    """Merges M sorted subarrays into a single sorted array."""
    import heapq
    merged_array = []
    
    # Use a min-heap to keep track of the smallest elements across subarrays
    min_heap = []

    # Push the first element of each subarray into the heap
    for i, subarray in enumerate(sorted_subarrays):
        if subarray:  # Ensure the subarray is not empty
            heapq.heappush(min_heap, (subarray[0], i, 0))

    # Merge process
    while min_heap:
        value, subarray_index, element_index = heapq.heappop(min_heap)
        merged_array.append(value)

        # If the subarray still has more elements, push the next element into the heap
        if element_index + 1 < len(sorted_subarrays[subarray_index]):
            next_value = sorted_subarrays[subarray_index][element_index + 1]
            heapq.heappush(min_heap, (next_value, subarray_index, element_index + 1))

    return merged_array

# Example Usage
if __name__ == "__main__":
    arr = [34, 7, 23, 32, 5, 62, 31, 8, 24, 19]
    m = 3  # Number of partitions for M-way merge sort
    print(f"Original Array: {arr}")
    sorted_arr = m_way_merge_sort(arr, m)
    print(f"Sorted Array: {sorted_arr}")
