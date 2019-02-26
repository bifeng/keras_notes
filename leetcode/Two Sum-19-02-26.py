'''
https://leetcode-cn.com/problems/two-sum/

'''


class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        out_ = []
        for j in range(len(nums)):
            for i in range(1, len(nums)):
                if nums[i] + nums[j] == target and nums[i] != nums[j]:
                    out_ = [i] + [j]
        out_.sort()
        return out_

# first version
def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    out_ = []
    for j in range(len(nums)):
        for i in range(1, len(nums)):
            if nums[i] + nums[j]==target and nums[i] != nums[j]:
                out_ = [i] + [j]
    out_.sort()
    return out_

twoSum([3,3], 6)
