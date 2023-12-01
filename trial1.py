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
file_path = 'adjusted_data.txt'
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

r_values = [2**8, 2**10, 2**12]
d_values = [1, 5, 10]
different_sketches = [process_stream_simulation_count_sketch, process_stream_simulation_med_sketch, process_stream_simulation_min_sketch]
window_sizes = [100, 200, 500]

def main():
    results = {}  # Dictionary to store results

    for r_val in r_values:
        for d_val in d_values:
            for sketch_process in different_sketches:
                total_accuracy = 0
                count = 0

                for window_size in window_sizes:
                    accuracy = sketch_process(all_urls=urls, window_size=window_size, d=d_val, r=r_val)
                    total_accuracy += sum(accuracy)
                    count += len(accuracy)

                average_accuracy = total_accuracy / count if count != 0 else 0
                key = (r_val, d_val, sketch_process.__name__)
                results[key] = average_accuracy

    return results

def plot_results(results):
    r_values, d_values, sketch_names, accuracies = zip(*[(r, d, sketch, acc) for (r, d, sketch), acc in results.items()])
    fig, ax = plt.subplots()
    for sketch in set(sketch_names):
        sketch_accuracies = [acc for r, d, name, acc in zip(r_values, d_values, sketch_names, accuracies) if name == sketch]
        ax.plot(sketch_accuracies, label=sketch)

    ax.set_xlabel('Combination Index')
    ax.set_ylabel('Average Accuracy')
    ax.set_title('Average Accuracy by Sketch Type, r Value, and d Value')
    ax.legend()

    plt.show()


results=main()
plot_results(results)


# def count_dict_binary_search(count_dic, query_count):
#     sorted_counts = sorted(count_dic.items(), keys=lambda x: x[1])

#     left, right = 0, len(sorted_counts) - 1
#     while left <= right:
#         mid = left + (right - left) // 2

#count_dict = {'id1': 10, 'id2': 20, 'id3': 30, 'id4': 40}
#query_count = 26
#closest_id = find_closest_id(count_dict, query_count)
#print(f"The ID with the count closest to {query_count} is {closest_id}.")