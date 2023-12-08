from sklearn.utils import murmurhash3_32
from bitarray import bitarray
import mmh3
import math
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from collections import Counter
from urllib.parse import urlparse

class BloomFilter:
    def __init__(self, n, fp_rate):
        self.n = n
        self.m = (math.ceil(-(self.n * math.log(fp_rate))/(math.log(2)**2)))
        self.k = int((self.m/self.n) * math.log(2))
        self.bit_array = bitarray(self.m)
        self.bit_array.setall(0)
    
    def insert(self, key):
        for i in range(self.k):
            index = murmurhash3_32(key, i) % self.m
            self.bit_array[index] = 1

    def test(self, key):
        for i in range(self.k):
            index = murmurhash3_32(key, i) % self.m
            if not self.bit_array[index]:
                return False
        return True
    
class CountMinSketch:
    def __init__(self, d, R):
        self.d = d 
        self.R = R 
        self.table = np.zeros((d, R), dtype=int)
        
    def insert(self, item):
        for i in range(self.d):
            j = mmh3.hash(item, i) % self.R
            self.table[i][j] += 1
            
    def query(self, item):
        cur_min = float('inf')
        for i in range(self.d):
            j = mmh3.hash(item, i) % self.R
            if self.table[i][j] < cur_min:
                cur_min = self.table[i][j]
        return cur_min

class CountMedianSketch:

    def __init__(self, d, R):
        self.d = d
        self.R = R 
        self.table = np.zeros((d, R), dtype=int)
        
    def insert(self, item):
        for i in range(self.d):
            j = mmh3.hash(item, i) % self.R
            self.table[i][j] += 1

    def query(self, item):
        counts = []
        for i in range(self.d):
            j = mmh3.hash(item, i) % self.R
            counts.append(self.table[i][j])
        return int(np.median(counts))
    
class CountSketch:
    def __init__(self, d, R):
        self.d = d
        self.R = R
        self.table = np.zeros((d, R), dtype=int)

    def insert(self, item):
        for i in range(self.d):
            j = mmh3.hash(item, i) % self.R
            sign = 1 if hash(item) % 2 == 0 else -1 
            self.table[i][j] += sign

    def query(self, item):
        counts = []
        for i in range(self.d):
            j = mmh3.hash(item, i) % self.R
            sign = 1 if hash(item) % 2 == 0 else -1
            counts.append(self.table[i][j] * sign)
        return int(np.median(counts))

# extracting data from file
file_path = 'dataset3.txt'
with open(file_path, 'r') as file:
    data_contents = file.read()
elems = data_contents.split('\n')

# parse out URLs
url_pattern = re.compile(r'\d+\. (.+)')
urls = []

# get the main part of URL from the data
for line in elems:
    whole_url = line.split(". ")[1]
    extracted_url = '/'.join(whole_url.split('/')[:3])
    urls.append(extracted_url)

# function to put WINDOW elems into BF. 
bloom_filter = BloomFilter(len(urls) - 1, 0.01)
for i in range(len(urls) - 1):
    pair = urls[i] + " " +  urls[i + 1]
    bloom_filter.insert(pair)

# Function to maintain window of 'window_size' and go through all URLs
# We check for every ith URL, where i > window_size - 1, the i+1th URL
# We verify whether (data_stream[i], prediction[i + 1]) exists in Bloom Filter
# We return accuracy score

r_values = [2**8, 2**10, 2**12]
d_values = [1, 5, 10]

def binary_search(count_dict, query_count):
    sorted_counts = sorted(count_dict.items(), key=lambda x: x[1])

    left, right = 0, len(sorted_counts) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if sorted_counts[mid][1] == query_count:
            return sorted_counts[mid][0]
        elif sorted_counts[mid][1] < query_count:
            left = mid + 1
        else:
            right = mid - 1

    if right < 0:  
        return sorted_counts[0][0]
    if left >= len(sorted_counts): 
        return sorted_counts[-1][0]

    if abs(sorted_counts[left][1] - query_count) < abs(sorted_counts[right][1] - query_count):
        return sorted_counts[left][0]
    else:
        return sorted_counts[right][0]


"""
We want to Vary on:
    1) Window Size
    2) D
    3) R
    4) Sketch type
    5) Number of elems we store in a pair
"""


def process_stream_simulation_min_sketch(all_urls, window_size, d, r):

    results = []

    for x in range(0, len(all_urls), window_size): 
        urls = all_urls[x: x + window_size]
        true_counts = Counter(urls)
        url_to_cs = {}
        count = 0
        
        # should be set urls I think
        for url in urls:
            c_min = CountMinSketch(d, r)
            for index in range(0, len(urls) - 1):
                if urls[index] == url:
                    c_min.insert(urls[index + 1]) 
            url_to_cs[url] = c_min

        for url in urls:
            query = url_to_cs[url].query(url)
            next_elem = binary_search(true_counts, query)
            if bloom_filter.test(url + " " + next_elem): 
                count += 1

        results.append(count/len(urls))

    return results

def process_stream_simulation_med_sketch(all_urls, window_size, d, r):

    results = []

    for x in range(0, len(all_urls), window_size): 
        urls = all_urls[x: x + window_size]
        true_counts = Counter(urls)
        url_to_cs = {}
        count = 0
        
        for url in urls:
            c_med = CountMedianSketch(d, r)
            for index in range(0, len(urls) - 1):
                if urls[index] == url:
                    c_med.insert(urls[index + 1]) 
            url_to_cs[url] = c_med

        for url in urls:
            query = url_to_cs[url].query(url)
            next_elem = binary_search(true_counts, query)
            if bloom_filter.test(url + " " + next_elem): 
                count += 1

        results.append(count/len(urls))

    return results

def process_stream_simulation_count_sketch(all_urls, window_size, d, r):

    results = []

    for x in range(0, len(all_urls), window_size): 
        urls = all_urls[x: x + window_size]
        true_counts = Counter(urls)
        url_to_cs = {}
        count = 0
        
        for url in urls:
            c_s = CountSketch(d, r)
            for index in range(0, len(urls) - 1):
                if urls[index] == url:
                    c_s.insert(urls[index + 1]) 
            url_to_cs[url] = c_s

        for url in urls:
            query = url_to_cs[url].query(url)
            next_elem = binary_search(true_counts, query)
            if bloom_filter.test(url + " " + next_elem): 
                count += 1

        results.append(count/len(urls))

    return results

r_values = [2**4, 2**6, 2**8]
d_values = [1, 5, 10]
window_sizes = [100, 200, 500]

def main():
    results = {}  # Dictionary to store results

    for r_val in r_values:
        for d_val in d_values:
            for window_size in window_sizes:
                accuracy = process_stream_simulation_count_sketch(all_urls=urls, window_size=window_size, d=d_val, r=r_val)
                
                key = (r_val, d_val, window_size)
                results[key] = accuracy

    return results

results = main()

########## ACCURACY VS CHANGING EACH PARAM ############
# Extracting data for the plots
# For r_val
accuracy_r = [sum(results.get((r, d_values[1], window_sizes[1]), [])) / len(results.get((r, d_values[1], window_sizes[1]), [])) for r in r_values]

# For d_val
accuracy_d = [sum(results.get((r_values[1], d, window_sizes[1]), [])) / len(results.get((r_values[1], d, window_sizes[1]), [])) for d in d_values]

# For window_size
accuracy_window = [sum(results.get((r_values[1], d_values[1], ws), [])) / len(results.get((r_values[1], d_values[1], ws), [])) for ws in window_sizes]

print(accuracy_r)
print(accuracy_d)
print(accuracy_window)


# Plotting
plt.figure(figsize=(15, 5))

# Plot for r_val
plt.subplot(1, 3, 1)
plt.plot(r_values, accuracy_r, marker='o')
plt.title('Accuracy vs r_val')
plt.xlabel('r_val')
plt.ylabel('Accuracy')

# Plot for d_val
plt.subplot(1, 3, 2)
plt.plot(d_values, accuracy_d, marker='o', color='green')
plt.title('Accuracy vs d_val')
plt.xlabel('d_val')

# Plot for window_size
plt.subplot(1, 3, 3)
plt.plot(window_sizes, accuracy_window, marker='o', color='red')
plt.title('Accuracy vs window_size')
plt.xlabel('window_size')
plt.savefig(f"accuracy_vary_param.png")
plt.tight_layout()
plt.show()

############ ACCURACY FOR EACH WINDOW FOR EACH COMBINATION OF PARAMS

# Setting up the plot
fig, axs = plt.subplots(9, 3, figsize=(15, 30))

for i, ((r_val, d_val, window_size), accuracies) in enumerate(results.items()):
    row = i // 3
    col = i % 3
    ax = axs[row, col]
    
    # Plotting the bar graph
    ax.bar(range(1, len(accuracies) + 1), accuracies)
    ax.set_title(f"r={r_val}, d={d_val}, window={window_size}")
    ax.set_xlabel("Window #")
    ax.set_ylabel("Accuracy")

plt.tight_layout()
plt.show()

############# SINGULAR PLOT OF ACCURACIES ACROSS WINDOW FOR A SPECIFIC KEY #################

for r in r_values:
    for d in d_values:
        for w in window_sizes:
            key = (r, d, w)
            # Check if the key exists in the results
            if key not in results:
                print(f"No results found for r={key[0]}, d={key[1]}, window={key[2]}")
            else:
                # Retrieve accuracies for the specified key
                accuracies = results[key]
                # Create a new figure for the plot
                plt.figure(figsize=(10, 6))
                # Plotting the bar graph
                plt.bar(range(1, len(accuracies) + 1), accuracies)
                plt.title(f"Accuracy for r={key[0]}, d={key[1]}, window={key[2]}")
                plt.xlabel("Window #")
                plt.ylabel("Accuracy")
                plt.savefig(f"accuracy_r{key[0]}_d{key[1]}_window{key[2]}.png")

                # Show the plot
                plt.tight_layout()
                #plt.show()
