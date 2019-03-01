'''
https://leetcode.com/problems/3sum-closest/
https://github.com/csujedihy/lc-all-solutions/blob/master/016.3sum-closest/3sum-closest.py

双指针 - 控制指针，易于遍历

'''

from typing import *


# first submission with bugs
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        closest_dev = None
        closest_sum = None
        for i in range(len(nums)):
            j, k = i + 1, len(nums) - 1

            while j < k:
                s = nums[i] + nums[j] + nums[k]

                dev = abs(s-target)

                if closest_dev is None:
                    closest_dev = dev
                    closest_sum = s
                    # use the first one to initial value

                elif dev > closest_dev:
                    k -= 1
                else:    # dev <= closest_dev
                    closest_dev = dev
                    closest_sum = s

                    # j += 1 or k -= 1  # 无论选择j += 1，还是k -= 1，都会忽略掉合理的值

        return closest_sum


# second
# Runtime  120 ms
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        closest_dev = None
        closest_sum = None
        for i in range(len(nums)):
            j, k = i + 1, len(nums)-1

            while j < k:
                s = nums[i] + nums[j] + nums[k]

                if s > target:
                    k -= 1
                elif s < target:
                    j += 1
                else:
                    return s

                dev = abs(s-target)

                if closest_dev is None:
                    closest_dev = dev
                    closest_sum = s
                    # use the first one to initial value

                if dev < closest_dev:
                    closest_dev = dev
                    closest_sum = s

        return closest_sum


# original code - very clear to express idea
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        ans = 0
        diff = float("inf")
        for i in range(0, len(nums)):
            start, end = i + 1, len(nums) - 1
            while start < end:
                sum = nums[i] + nums[start] + nums[end]
                if sum > target:
                    if abs(target - sum) < diff:
                        diff = abs(target - sum)
                        ans = sum
                    end -= 1
                else:
                    if abs(target - sum) < diff:
                        diff = abs(target - sum)
                        ans = sum
                    start += 1
        return ans


# simplify code - reduce some duplicate compute
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        ans = 0
        diff = float("inf")
        for i in range(0, len(nums)):
            start, end = i + 1, len(nums) - 1
            while start < end:
                sum = nums[i] + nums[start] + nums[end]
                if sum > target:
                    end -= 1
                elif sum < target:
                    start += 1
                else:
                    return sum

                if abs(target - sum) < diff:
                    diff = abs(target - sum)
                    ans = sum
        return ans


tt= Solution()
nums,target = [0,2,1,-3] , 1         # [-3,0,1,2]
# nums, target = [1,1,1,0], -100
# nums, target = [1,1,-1,-1,3], -1   # [-1,-1,1,1,3]
tt.threeSumClosest(nums=nums,target=target)

