'''
https://leetcode.com/problems/4sum/
https://github.com/csujedihy/lc-all-solutions/blob/master/018.4sum/4sum.py


1, 双层循环 + 双指针
2, 哈希
First sort num, then build a dictionary d, d[num[p]+num[q]] = [(p,q) pairs which satisfy num[p] + num[q]],
here all (p,q) pairs satisfy p < q.
Then use a nested for-loop to search, num[i] is the min number in quadruplet and num[j] is the second min number.
The time complexity of checking whether d has the key target - num[i] - num[j] is O(1).
If this key exists, add one quadruplet to the result. Use set() to remove duplicates in res,
otherwise for input [-3,-2,-1,0,0,1,2,3], 0 there will be two [-3, 0, 1, 2] and two [-2, -1, 0, 3].
http://chaoren.is-programmer.com/posts/45308.html
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


# second method  too slow and still have some bugs
# 1, hash
# 基本思路：将四数求和转化为两数求和问题
import itertools
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()

        # 两两组合
        two_num_combs = [i for i in itertools.combinations(nums, 2)]

        # hash
        number = {}  # 因为有些元组的和值可能相同，所以元组作为key，和值作为value。
        result = []
        for i in range(len(two_num_combs)):
            if target - sum(two_num_combs[i]) in number.values():
                multi_keys = get_key(number,target - sum(two_num_combs[i]))
                for j in multi_keys:
                    new_list = list(two_num_combs[i]) + list(j)
                    new_list.sort()
                    if new_list not in result and not_duplicate(new_list, nums):
                        result.append(new_list)
            else:
                number[two_num_combs[i] ] = sum(two_num_combs[i])
        return result


import collections
def not_duplicate(ls, nums):
    ls_dict = collections.Counter(ls)
    nums_dict = collections.Counter(nums)
    value = 1
    for key,value in ls_dict.items():
        if key in nums_dict:
            if value > nums_dict[key]:  # 某数字出现次数多于原始列表中该数字的出现次数
                return 0
    return value


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


# second original method
# Runtime 148 ms
# 1, hash
# key-两数之和，value-元素索引
# 通过索引控制，避免元素重复出现.
class Solution:
    # @return a list of lists of length 4, [[val1,val2,val3,val4]]
    def fourSum(self, num, target):
        numLen, res, dict = len(num), set(), {}
        if numLen < 4: return []
        num.sort()
        for p in range(numLen):
            for q in range(p+1, numLen):
                if num[p]+num[q] not in dict:
                    dict[num[p]+num[q]] = [(p,q)]
                else:
                    dict[num[p]+num[q]].append((p,q))  # 如果key重复，则新增到value
        for i in range(numLen):
            for j in range(i+1, numLen-2):
                T = target-num[i]-num[j]
                if T in dict:
                    for k in dict[T]:
                        if k[0] > j: res.add((num[i],num[j],num[k[0]],num[k[1]]))
        return [list(i) for i in res]


# third methods
# Runtime 108ms
# 基本思路：将多数求和不断转化为两数求和
# Good Solution!
class Solution:
    def fourSum(self, nums, target):
        def findNsum(nums, target, N, result, results):

            # early termination (1-数字不够,2-第一个太大,3-最后一个太小)
            if N < 2 or len(nums) < N or target < nums[0]*N or target > nums[-1]*N:
                return

            # two pointers solve sorted 2-sum problem
            if N == 2:
                l,r = 0,len(nums)-1
                while l < r:
                    s = nums[l] + nums[r]
                    if s == target:
                        results.append(result + [nums[l], nums[r]])

                        # todo 为什么是l-1而不是l+1?
                        # https://leetcode.com/problems/4sum/discuss/8545/Python-140ms-beats-100-and-works-for-N-sum-(Ngreater2)
                        # skip the duplicates
                        while l < r and nums[l] == nums[l-1]:
                            l += 1
                        while l < r and nums[r] == nums[r+1]:
                            r -= 1

                        # while l < r and nums[l] == nums[l+1]:
                        #     l += 1
                        # while l < r and nums[r] == nums[r-1]:
                        #     r -= 1

                        l += 1
                        r -=1

                    elif s < target:
                        l += 1
                    else:
                        r -= 1

            else:
                # recursively reduce N
                for i in range(len(nums)-N+1):
                    if i == 0 or (i > 0 and nums[i] != nums[i-1]):
                        findNsum(nums[i+1:], target-nums[i], N-1, result+[nums[i]], results)

        nums.sort()
        results = []
        findNsum(nums, target, 4, [], results)
        return results


tt = Solution()
nums, target = [1, 0, -1, 0, -2, 2], 0
# output [[-2, -1, 1, 2], [-1, -1, 0, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]
# output [[-2, -1, 1, 2], [-2, 0, 0, 2]]
# expect [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]
# nums, target = [-3,-2,-1,0,0,1,2,3],0
# output [[-3, -2, 2, 3],[-3, -1, 1, 3], [-2, -1, 0, 3], [-3, 0, 0, 3], [-1, 0, 0, 1], [-2, 0, 0, 2], [-2, -1, 1, 2]]
# output [[-3, -2, 2, 3], [-3, -1, 1, 3], [-3, 0, 0, 3], [-3, 0, 1, 2]]
# expect [[-3,-2,2,3],[-3,-1,1,3],[-3,0,0,3],[-3,0,1,2],[-2,-1,0,3],[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
# nums, target = [0,2,2,2,10,-3,-9,2,-10,-4,-9,-2,2,8,7], 6
# output [[-9,-3,8,10],[-9,-2,7,10],[-10,-2,8,10],[-4,-2,2,10],[-4,0,2,8],[-3,0,2,7],[-9,0,7,8]]
# expect [[-10,-2,8,10],[-9,-3,8,10],[-9,-2,7,10],[-9,0,7,8],[-4,-2,2,10],[-4,0,2,8],[-3,0,2,7],[0,2,2,2]]
print(tt.fourSum(nums,target))

