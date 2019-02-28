'''
https://leetcode.com/problems/3sum-closest/

'''

from typing import *

class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        closest_dev = None
        closest_sum = None
        for i in range(len(nums)):
            j, k = i + 1, len(nums) - 1

            while j < k:
                s = nums[i] + nums[j] + nums[k]
                dev = abs(s - target)

                if closest_dev is None:
                    closest_dev = dev
                    closest_sum = s

                    j += 1
                    k -= 1
                elif dev > closest_dev:
                    k -= 1
                elif dev < closest_dev:
                    closest_dev = dev
                    closest_sum = s

                    j += 1
                else:
                    closest_dev = dev
                    closest_sum = s
                    j += 1
                    k -= 1

        return closest_sum

[0,2,1,-3]  1
[1,1,1,0] -100
