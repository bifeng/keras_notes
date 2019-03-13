'''
https://leetcode.com/problems/sort-colors/
https://leetcode.com/problems/sort-colors/discuss/26481/Python-O(n)-1-pass-in-place-solution-with-explanation

1. dutch partitioning problem

2. move zeroes

'''

from typing import *


# first method
# 借鉴Move Zeroes思想
# - 从大到小依次替换
# - 记录第一个数字位置，若当前数字小于该数，则交换数值，该数位置加1
# Runtime 40ms
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for j in [2, 1]:
            lastindex = 0
            for i, v in enumerate(nums):
                if v < j:
                    nums[lastindex], nums[i] = nums[i], nums[lastindex]
                    lastindex += 1


# second method
# runtime 44 ms
# Good Solution!
'''
This is a dutch partitioning problem. 
We are classifying the array into four groups: red, white, unclassified, and blue. 
Initially we group all elements into unclassified. 
We iterate from the beginning as long as the white pointer is less than the blue pointer.
Imagine red and blue are separators for the three sections, and white is scan pointer.
'''
class Solution:
    def sortColors(self, nums):
        red, white, blue = 0, 0, len(nums) - 1

        while white <= blue:
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red]
                white += 1
                red += 1
            elif nums[white] == 1:
                white += 1
            else:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1


