'''
https://leetcode.com/problems/3sum/
https://leetcode.com/problems/3sum/discuss/232712/Best-Python-Solution-(Explained)
双指针

'''
from typing import *


# first method 超时!
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()
        for i in range(len(nums)):
            target = nums[i]
            new_nums = nums[:i] + nums[(i+1):]
            j, k = 0, len(new_nums)-1
            while j < k:
                s = new_nums[j] + new_nums[k]
                if s + target > 0:
                    k += -1
                elif s + target < 0:
                    j += 1
                else:
                    result.append([nums[i],new_nums[j],new_nums[k]])
        return result


# second method
# Runtime  848 ms
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()
        for i in range(len(nums)-2):  # 剩余的两个数不需要考虑，因为已经考虑过
            if nums[i] > 0: break  # 如果当前这个数大于0，后面三个数之和也会大于0，不需要考虑
            if i > 0 and nums[i] == nums[i-1]: continue  # 如果当前这个考虑过，现在也不需要考虑

            j, k = i+1, len(nums)-1  # 如果i之前不满足，之后也不需要考虑，所以从i+1开始
            while j < k:
                s = nums[j] + nums[k]
                if s + nums[i] > 0:
                    k += -1
                elif s + nums[i] < 0:
                    j += 1
                else:
                    result.append([nums[i],nums[j],nums[k]])

                    # 当前考虑过的数字，下次也不需要考虑
                    while j < k and nums[j] == nums[j+1]:
                        j += 1
                    while j < k and nums[k] == nums[k-1]:
                        k += -1

                    # Because you need to skip two identical numbers , if it exists
                    j += 1
                    k += -1
        return result


tt = Solution()
resul = tt.threeSum([-2,0,0,2,2])
print(resul)