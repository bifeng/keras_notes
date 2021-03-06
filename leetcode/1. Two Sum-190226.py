'''
https://leetcode.com/problems/two-sum/solution/
1, hash map
-哈希值判断另外一个数是否存在

2, 双指针
-双指针从左右两边向中间遍历
-但是需要copy原始列表，以提取原始index
-还需要从列表提取数值相等的两个数字index

'''

# first method
# Runtime 6108 ms
def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    out_ = []
    for j in range(len(nums)):
        for i in range(j+1, len(nums)):  # 遍历j后面的每个元素 good!
            if nums[i] + nums[j]==target:
                out_ = [i] + [j]
    return out_


# second method
# python - dict <=> java - HashMap

# 字典是通过hash表的原理实现的，每个元素都是一个键值对，通过元素的键计算出一个唯一的哈希值，这个hash值决定了元素的地址，因此为了保证元素地址不一样，必须保证每个元素的键和对应的hash值是完全不同的，并且键的类型必须是不可修改的，所以键的类型可以使数值，字符串常量或元组，但不能是列表，因为列表是可以被修改的。
#
# 所以字典具有下列特性：
# 1、元素的查询和插入操作很快，基本上是常数级别
# 2、占用内存较大，采用的是空间换时间的方法


# Runtime 56 ms
# 1, hash
# key-元素，value-元素索引
def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    sum_value = {}
    for j in range(len(nums)):
        if target - nums[j] in sum_value.keys():
            return [sum_value[target-nums[j]], j]
        else:
            sum_value[nums[j]] = j


# Runtime 56 ms
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        new_nums = nums.copy()
        new_nums.sort()
        i, j = 0, len(new_nums) - 1

        while i < j:
            s = new_nums[i] + new_nums[j]

            if s > target:
                j -= 1
            elif s < target:
                i += 1
            else:

                if new_nums[i] == new_nums[j]:
                    result = [index for index,v in enumerate(nums) if v == new_nums[i]]
                else:
                    result = [nums.index(new_nums[i]), nums.index(new_nums[j])]
                result.sort()
                return result


tt = Solution()
# nums, target = [3,3], 6
nums, target = [3,2,3], 6
# nums, target = [3,2,4], 6
tt.twoSum(nums, target )

