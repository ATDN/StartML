a=int(input())
if(a<0):
    print("Invalid Value!")
elif(a<=50):
    print("cost = {:.2f}".format(a*0.53))
else:
    print("cost = {:.2f}".format(50*0.53+0.58*(a-50)))