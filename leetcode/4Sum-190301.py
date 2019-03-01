'''
https://leetcode.com/problems/4sum/
https://github.com/csujedihy/lc-all-solutions/blob/master/018.4sum/4sum.py


1, 双层循环 + 双指针
2, 哈希
https://www.cnblogs.com/zuoyuan/p/3699384.html
https://github.com/chaor/LeetCode_Python_Accepted
3, 递归 + 双指针
https://leetcode.com/problems/4sum/discuss/8545/Python-140ms-beats-100-and-works-for-N-sum-(Ngreater2)


'''
from typing import *


# first method
# Runtime 1112 ms
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()

        result = []
        for i in range(len(nums) - 3):
            for j in range(i + 1, len(nums)):

                start, end = j + 1, len(nums) - 1

                while start < end:
                    sum = nums[i] + nums[j] + nums[start] + nums[end]

                    if sum < target:
                        start += 1
                    elif sum > target:
                        end -= 1
                    else:
                        new_list = [nums[i], nums[j], nums[start], nums[end]]
                        if new_list not in result:
                            result.append(new_list)

                        start += 1
                        end -= 1
        return result


# original method
# Runtime 1116 ms
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        nums.sort()
        res = []
        for i in range(0, len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(i + 1, len(nums)):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                start, end = j + 1, len(nums) - 1
                while start < end:
                    sum = nums[i] + nums[j] + nums[start] + nums[end]
                    if sum < target:
                        start += 1
                    elif sum > target:
                        end -= 1
                    else:
                        res.append((nums[i], nums[j], nums[start], nums[end]))
                        start += 1
                        end -= 1

                        # it is weird ?
                        while start < end and nums[start] == nums[start - 1]:
                            start += 1
                        while start < end and nums[end] == nums[end + 1]:
                            end -= 1
        return res

