from sklearn.utils import murmurhash3_32
from bitarray import bitarray
import math
import pandas as pd
import matplotlib.pyplot as plt
import re


def next_power_of_two(n):
    power = 1
    while power < n:
        power *= 2
    return power

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

# storing 2 element pairs in bf
file_path = 'adjusted_data.txt'
with open(file_path, 'r') as file:
    data_contents = file.read()
elems = data_contents.split('\n')

# gpt wrote this for me
url_pattern = re.compile(r'\d+\. (.+)')
urls = [url_pattern.match(line).group(1) for line in elems if url_pattern.match(line)]

bloom_filter = BloomFilter(len(urls) - 1, 0.01)
for i in range(len(urls) - 1):
    pair = urls[i] + " " +  urls[i + 1]
    bloom_filter.insert(pair)

# testing
print(bloom_filter.test("chrome://newtab/ chrome://newtab/"))
