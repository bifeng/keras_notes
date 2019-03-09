'''
https://leetcode.com/problems/move-zeroes/
https://leetcode.com/problems/move-zeroes/solution/

todo
traverse + swap
'''


from typing import *


# first method
# create another array
# Runtime 64ms
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        a = [v for i, v in enumerate(nums) if v != 0]
        len_ = len(nums) - len(a)
        a += [0 for i in range(len_)]

        for i, v in enumerate(a):
            nums[i] = v


# second method
# 基本思路：
# 关注0，将0与非0交换
# 关注非0，将非0与0交换
# 借助一个指针记录0或非0元素的位置
# Runtime  48ms
# Good Solution!
class Solution:
    def moveZeroes(self, nums):
        zero = 0  # records the position of "0"
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[zero] = nums[zero], nums[i]
                zero += 1


ts = Solution()
nums = [0,1,0,3,12]
print('merge result:',ts.moveZeroes(nums))


