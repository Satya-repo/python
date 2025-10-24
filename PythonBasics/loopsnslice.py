for x in range(100):
    print('printing square ',x*x)

items = ['satya','bangalore','maldives','tanima']

for item in items:
    if item == 'bangalore':
        print('printing city now ',item)

for i in range(1, 6):
    print(f"  {i}")

print("\nEven numbers from 2 to 10:")
for i in range(2, 11, 2):  # start, stop, step
    print(f"  {i}")


# 5. For loop with dictionaries

student_grades = {'Alice': 85, 'Bob': 92, 'Charlie': 78, 'Diana': 96}
print("Student grades:")
for name, grade in student_grades.items():
    print(f"  {name}: {grade}")


# 8. For loop with else clause
print("Searching for number 3 in range 1-5:")
for i in range(1, 6):
    if i == 3:
        print(f"  Found {i}!")
        break
else:
    print("  Number 3 not found!")

# 9. List comprehension (for loop in one line)

squares = [x**2 for x in range(1, 6, 2)]
print(f"Squares: {squares}")

cube = [x*x*x for x in range(2,10,2) if x%2==0]
print('printing cube now ',cube)

even_numbers = [x for x in range(1, 11) if x % 2 == 0]
print(f"Even numbers: {even_numbers}")


# 1. Basic while loop

count = 1
while count <= 5:
    print(f"  {count}")
    count += 1


# 3. While loop with break
print("\n3. WHILE LOOP WITH BREAK")
number = 1
print("Numbers 1-10, stop at 7:")
while True:
    if number > 10:
        break
    if number == 7:
        print(f"  {number} - Stopping here!")
        break
    print(f"  {number}")
    number += 1

# 4. While loop with continue
print("\n4. WHILE LOOP WITH CONTINUE")
number = 0
print("Even numbers from 2 to 10:")
while number < 10:
    number += 1
    if number % 2 != 0:
        continue  # Skip odd numbers
    print(f"  {number}")


# Now lets work with slice

item = [x*x for x in range(2,20)]
print(item)
print('now checking slice ',item[2:4:2])