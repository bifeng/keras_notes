'''
https://leetcode.com/problems/merge-sorted-array/


'''

from typing import *


# first method
# 基本思路：遍历nums2，查找nums1中与之最近的数字及其索引，根据索引插入
# class Solution:
#     def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
#         """
#         Do not return anything, modify nums1 in-place instead.
#         """
#         j = 0
#         start, end = 0, n - 1
#
#         min_dif = 0
#         while start < end:
#             if nums2[start] - nums1[j] > min_dif:



# second method
# 基本思路：todo 这是什么思维？还需要多消化...
# https://leetcode.com/problems/merge-sorted-array/discuss/29503/Beautiful-Python-Solution
# https://leetcode.com/problems/merge-sorted-array/discuss/29522/This-is-my-AC-code-may-help-you
# Good Solution!
class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        Do not return anything, modify nums1 in-place instead.
        """
        while m > 0 and n > 0:
            if nums1[m - 1] >= nums2[n - 1]:
                nums1[m + n - 1] = nums1[m - 1]
                m -= 1
            else:
                nums1[m + n - 1] = nums2[n - 1]
                n -= 1
        if n > 0:
            nums1[:n] = nums2[:n]


ts = Solution()
nums1,m,nums2,n = [1,2,3,0,0,0] ,3 ,[2,5,6] ,3
print('merge result:',ts.merge(nums1,m,nums2,n))
