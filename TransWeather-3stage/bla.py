class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        s1p = 0
        s2p = 0
        s3p = 0

        # oscillate between s1 and s2 and try to get as many symbols
        order = None
        while s3p < len(s3):
            if order:
                pass
            else:
                pass
            '''
            if s1p < len(s1) and c == s1[s1p]:
                s1p += 1
            elif s2p < len(s2) and c == s2[s2p]:
                s2p += 1
            else:
                return False
            '''
        return True

if __name__ == '__main__':
    s = Solution()
    s1 = "aabcc"
    s2 = "dbbca"
    s3 = "aadbbcbcac"
    print(s.isInterleave(s1, s2, s3))