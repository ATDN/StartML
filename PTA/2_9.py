#map把所有的都变成int，然后整体变为list
lst=list(map(int,input().split()))
lst.sort()
lst=[str(i) for i in lst]
#将序列中的元素用指定符号连接成一个字符串输出
print('->'.join(lst))