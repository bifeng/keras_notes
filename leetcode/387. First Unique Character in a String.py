'''
https://leetcode.com/problems/first-unique-character-in-a-string/


'''


# first method
# Runtime  208 ms
class Solution:
    def firstUniqChar(self, s: str) -> int:
        index = -1
        hashdict = {}
        for i,v in enumerate(s):
            if v not in hashdict.keys():
                hashdict[v] = [i]
            else:
                hashdict[v].append(i)
        print(hashdict)
        for key,val in hashdict.items():
            if len(val)==1:
                return val[0]
        return index

