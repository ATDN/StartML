a=int(input())
b=sum(1/i for i in range(1,2*a+1,2))
print("sum = {:.6f}".format(b))