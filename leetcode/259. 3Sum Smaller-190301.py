'''
https://leetcode.com/problems/3sum-smaller - locked

https://github.com/csujedihy/lc-all-solutions/blob/master/259.3sum-smaller/question.md
https://github.com/csujedihy/lc-all-solutions/blob/master/259.3sum-smaller/3sum-smaller.py

Given an array of n integers nums and a target, find the number of index triplets i, j, k with 0 <= i < j < k < n that satisfy the condition nums[i] + nums[j] + nums[k] < target.
For example, given nums = [-2, 0, 1, 3], and target = 2.
Return 2. Because there are two triplets which sums are less than 2:
[-2, 0, 1] [-2, 0, 3]
Follow up: Could you solve it in O(n2) runtime?

1, 双指针
- 如果只是计数，可以通过满足条件的这个区间组合计数。避免直接去获取符合条件的列表
-
'''

from typing import *


# second
class Solution(object):
  def threeSumSmaller(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    ans = 0
    nums.sort()
    for i in range(0, len(nums)):
      start, end = i + 1, len(nums) - 1
      while start < end:
        if nums[i] + nums[start] + nums[end] < target:
          ans += end - start
          start += 1
        else:
          end -= 1

    return ans


# first submission with bugs
class Solution:
    def threeSumSmaller(self, nums: List[int], target: int) -> int:
        nums.sort()
        count = 0

        for i in range(len(nums)):
            l, k = i+1, len(nums)-1

            while l < k:
                s = nums[i] + nums[l] + nums[k]

                if s < target:
                    l += 1    # l += 1 会忽略掉合理的值
                    count += 1
                else:
                    k -= 1
        return count


tt= Solution()
nums,target = [-2, 0, 1, 3] ,2
tt.threeSumSmaller(nums=nums,target=target)
