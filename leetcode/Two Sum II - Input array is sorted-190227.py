'''
https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/submissions/
1, hash map
2,
3,
'''
from typing import *

# first method
# Runtime 40 ms
def twoSum(nums: List[int], target: int) -> List[int]:
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    sum_value = {}
    for j in range(len(nums)):
        if target - nums[j] in sum_value.keys():
            return [sum_value[target - nums[j]] + 1, j + 1]
        else:
            sum_value[nums[j]] = j

# second method
def twoSum(nums: List[int], target: int) -> List[int]:
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    i = 0
    sum_value = {}
    for j in range(len(nums)):
        if target - nums[j] in sum_value.keys():
            return [sum_value[target - nums[j]] + 1, j + 1]
        else:
            sum_value[nums[j]] = j

