'''
https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/1/array/21/

+ 针对动态的list，如何控制指针？
  1) range遍历
  2) 双指针

+ 两种方式：删除 + 改写

+ 两种方式：判断相等 + 判断不相等

'''


# five version - 不相等则位置不变
def removeDuplicates(nums):
    if len(nums) == 0:
        return 0
    j = 0
    for i in range(1, len(nums)):
        if nums[i] != nums[j]:
            nums[j + 1] = nums[i]
            j = j + 1
    print(nums)
    return j + 1

# five version - 相等则位置往前移动一位 - todo wrong result?
def removeDuplicates(nums):
    if len(nums) == 0:
        return 0
    j = 0
    for i in range(1, len(nums)):
        if nums[i] == nums[j]:
            nums[j] = nums[i]
            j = j + 1
    return j + 1

#...

# first version
def removeDuplicates(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    for i, v in enumerate(nums):
        len_ = len(nums)
        if len_ > i+1 and nums[i+1] == v:
            del nums[i]
        if i-1>=0 and nums[i-1] == v:
            del nums[i-1]
    return len(nums)

removeDuplicates([1,1,2])
removeDuplicates([0,0,0,0,0])
removeDuplicates([1,1,1,1])
removeDuplicates([1,1,1])
removeDuplicates([1,1])
removeDuplicates([1])