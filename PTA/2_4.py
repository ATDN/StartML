a,n=input().split()
#先是字符*2，再转化为数字
b = sum([int(a*i) for i in range(1,int(n)+1)])
print("s = {:d}".format(b))