import bisect

class VBQ:
    def __init__(self, val, partition):
        self.__partition = partition
        self.__val = self.from_float(val)
        
    def to_float(self):
        if self.__val == 0:
            return self.__partition[0];
        if self.__val == len(self.__partition):
            return self.__partition[len(self.__partition) - 1]
        return (self.__partition[self.__val] + self.__partition[self.__val-1]) / 2

    def from_float(self, fl):
        return bisect.bisect_left(self.__partition, fl)


if __name__=='__main__':
    # test
    a = VBQ(-4.0, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    if a.to_float() == -3.0:
        print("PASS!")
    else:
        print("FALSE")

    a = VBQ(4.0, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    if a.to_float() == 3.0:
        print("PASS!")
    else:
        print("FALSE")

    a = VBQ(1.2, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    if a.to_float() == 1.5:
        print("PASS!")
    else:
        print("FALSE")

    a = VBQ(-2.1, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    if a.to_float() == -2.5:
        print("PASS!")
    else:
        print("FALSE")

    a = VBQ(2.7, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    if a.to_float() == 2.5:
        print("PASS!")
    else:
        print("FALSE")
