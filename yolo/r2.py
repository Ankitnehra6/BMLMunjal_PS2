def checkPalindrome(num):
    tem=num
    rev=0
    while(num>0):
        d=num%10
        rev=rev*10+d
        num=num//10
    if tem == rev:
        return True
    
num = int(input())
isPalindrome = checkPalindrome(num)
if (isPalindrome):
    print('true')
else:
    print('false')