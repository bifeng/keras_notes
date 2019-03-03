'''
https://leetcode.com/problems/contains-duplicate/
https://leetcode.com/problems/contains-duplicate/solution/

1, sort
Intuition
If there are any duplicate integers, they will be consecutive after sorting.

2, Binary Search Tree and Hash Table
Intuition
Utilize a dynamic data structure that supports fast search and insert operations.
There are many data structures commonly used as dynamic sets such as Binary Search Tree and Hash Table.
The operations we need to support here are search() and insert().

Note:
    One should keep in mind that real world performance can be different from what the Big-O notation says.
    The Big-O notation only tells us that for sufficiently large input, one will be faster than the other.
'''

from typing import *


# first methods
# Runtime  10816 ms
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums_list = []
        is_dup = False
        for i in range(len(nums)):
            if nums[i] in nums_list:
                return True
            else:
                nums_list.append(nums[i])
        return is_dup


# second methods
# sort
# Runtime  64ms
# Good Solution!
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums.sort()  # 加个排序后，重复元素查找更快
        is_dup = False
        for i in range(len(nums)):
            if i < len(nums) - 1:
                if nums[i] == nums[i + 1]:
                    return True
        return is_dup


# simplify
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums.sort()  # 加个排序后，重复元素查找更快
        is_dup = False
        for i in range(len(nums)-1):
            if nums[i] == nums[i + 1]:
               return True
        return is_dup


# third methods
# hash - dict
# Runtime 56 ms
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums_dict = {}
        is_dup = False
        for i in range(len(nums)):
            if nums[i] in nums_dict.keys():
                return True
            else:
                nums_dict[nums[i]] = i
        return is_dup


# third methods
# hash - set
# Runtime  52ms
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums_set = set()
        is_dup = False
        for i in range(len(nums)):
            if nums[i] in nums_set:
                return True
            else:
                nums_set.add(nums[i])
        return is_dup


# third methods
# hash - set
# Runtime 44 ms
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        if len(nums) != len(set(nums)):
            return True
        else:
            return False


tt = Solution()
nums, target = [1, 0, 3, 10, -2, 2], 0
print(tt.containsDuplicate(nums))
