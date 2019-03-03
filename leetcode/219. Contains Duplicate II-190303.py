'''
https://leetcode.com/problems/contains-duplicate-ii/
https://leetcode.com/problems/contains-duplicate-ii/discuss/61375/Python-concise-solution-with-dictionary.

1, hash + dict
2, hash + set
3, hash set + queue
'''

from typing import *


# first methods
# hash - dict
# key为数值，value为索引 - 该值的当前索引减去该值的前一个索引
# Runtime  72 ms
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        num_dict = {}
        dif_ = []
        for i in range(len(nums)):
            if nums[i] in num_dict.keys():
                num_dict[nums[i]].append(i)

                value = num_dict[nums[i]]
                dif_.append(value[-1] - value[-2])
            else:
                num_dict[nums[i]] = [i]

        if len(dif_):
            if min(dif_) <= k:
                return True
            else:
                return False
        else:
            return False


# simplify
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        num_dict = {}
        for i in range(len(nums)):
            if nums[i] in num_dict.keys():
                num_dict[nums[i]].append(i)

                value = num_dict[nums[i]]
                if value[-1] - value[-2] <= k:
                    return True

            else:
                num_dict[nums[i]] = [i]
        return False


# simplify
# Runtime 48ms
# Perfect Solutions!
class Solution:
    def containsNearbyDuplicate(self, nums, k):
        dic = {}
        for i, v in enumerate(nums):
            if v in dic and i - dic[v] <= k:
                return True
            dic[v] = i
        return False


# second method
# hash - set
# Runtime 48ms
# hash a K-length moving window of nums.
# When moving along, do not reconstruct the set,
# just add the new one, and remove the earliest element.
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        if len(nums)<=k:
            return len(nums) > len(set(nums))

        hashSet=set(nums[:k])
        if len(hashSet) < k:
            return True

        for i in range(k,len(nums)):
            hashSet.add(nums[i])
            if len(hashSet)==k:
                return True
            else:
                hashSet.remove(nums[i-k])
        return False


# second method
# hash - set
# Time Limit Exceeded todo why?
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        if len(nums) <= k:
            return len(nums) > len(set(nums))

        for i in range(len(nums)):
            if i >= k:
                if len(set(nums[(i - k):(i + 1)])) <= k:
                    return True
        return False


# other methods
# queue
# Runtime 68ms
# https://github.com/csujedihy/lc-all-solutions/blob/master/219.contains-duplicate-ii/contains-duplicate-ii.py
from collections import deque
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        if not nums:
            return False
        if k == 0:
            return False
        k = k + 1
        k = min(k, len(nums))

        window = deque([])
        d = set()
        for i in range(0, k):
            if nums[i] in d:
                return True
            d |= {nums[i]}
            window.append(i)
        for i in range(k, len(nums)):
            d -= {nums[window.popleft()]}
            if nums[i] in d:
                return True
            d |= {nums[i]}
            window.append(i)
        return False


tt = Solution()
nums, target = [1,0,1,1], 1
nums, target = [1,2,3,1], 3
nums, target = [99,99],2
print(tt.containsNearbyDuplicate(nums,target))

