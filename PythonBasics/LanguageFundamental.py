# There are 4 types of data types
#int, float, str, bool

print(type(2))
print(type(2.222))
print(type('satya'))
print(type(True))

# Basics of printing

print('hello world','satya')
print('hello world',4.5,end='|')
print('hello',end='\n')
print('hello again')

x = 'hello'
y = 'hello'
print(x==y)
x = 'satya'
print(x == y)
z = 'tanima'

# printing through f'String now

print(f"My name is {x} and not same as {z}")

print(f"{2+3} is actually greater then {1+1}")

print(f"Factorila of 5 is {60+60}")

print(f"My name is {'satya'}")

# Lets understand taking input from user

myname = input('Name ')
print('My name is', myname, 'and its coming from input prompt')

age =  input('Age is ')
print('My age is ',age)

# Lets accept list as input

items = input('input marks separated by comma ')
items = items.split(',')
print('marks given by student is ',items)

for x in items :
    print(x)
