# second version
def removeDuplicates(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    len_ = len(nums)
    print(len_)
    for i, v in enumerate(nums):
        try:
            if len_ > i+1 and nums[i+1] == v:
                del nums[i+1]

            if len(nums)!=1 and nums[i-1] == v:
                del nums[i-1]
        except:
            pass
    return len(nums)



# first version
def removeDuplicates(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    count = 0
    for i, v in enumerate(nums):
        if sum([j == v for j in nums]) >= 2:
            dup_index = [j for j, vs in enumerate(nums) if vs == v and j != i]
            dup_index.reverse()
            for i in dup_index:
                del nums[i]
            count +=1
        else:
            count += 1
    print(nums)
    return count


removeDuplicates([1,1,2])
removeDuplicates([1,1,1,1])
removeDuplicates([1,1,1])
removeDuplicates([1,1])
removeDuplicates([1])