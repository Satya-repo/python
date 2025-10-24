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
print("\n1. BASIC WHILE LOOP")
count = 1
print("Counting from 1 to 5:")
while count <= 5:
    print(f"  {count}")
    count += 1

# 2. While loop with user input
print("\n2. WHILE LOOP WITH USER INPUT SIMULATION")
print("Simulating user input (entering 'quit' to stop):")
user_input = "hello"
attempts = 0
while user_input != "quit" and attempts < 3:
    attempts += 1
    if attempts == 1:
        user_input = "hello"
    elif attempts == 2:
        user_input = "world"
    else:
        user_input = "quit"
    print(f"  Attempt {attempts}: {user_input}")

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

# 5. While loop with else clause
print("\n5. WHILE LOOP WITH ELSE CLAUSE")
count = 1
print("Counting 1-3:")
while count <= 3:
    print(f"  {count}")
    count += 1
else:
    print("  Loop completed normally!")

# 6. Infinite loop prevention
print("\n6. INFINITE LOOP PREVENTION")
counter = 0
max_attempts = 5
print("Loop with safety counter:")
while counter < max_attempts:
    counter += 1
    print(f"  Attempt {counter}")
    if counter >= 3:
        print("  Breaking early!")
        break

# 7. While loop for data validation
print("\n7. WHILE LOOP FOR DATA VALIDATION")
print("Simulating age validation:")
age = 0
while age < 18 or age > 100:
    age = 25  # Simulate valid input
    if age < 18:
        print(f"  Age {age} is too young!")
    elif age > 100:
        print(f"  Age {age} is too old!")
    else:
        print(f"  Age {age} is valid!")

# 8. While loop with multiple conditions
print("\n8. WHILE LOOP WITH MULTIPLE CONDITIONS")
temperature = 20
target_temp = 25
heating = True
print("Temperature control simulation:")
while temperature < target_temp and heating:
    temperature += 2
    print(f"  Temperature: {temperature}°C")
    if temperature >= target_temp:
        heating = False
        print("  Target temperature reached!")

# 9. While loop for menu system
print("\n9. WHILE LOOP FOR MENU SYSTEM")
print("Simulating menu system:")
choice = 1
menu_count = 0
while choice != 4 and menu_count < 3:
    menu_count += 1
    if menu_count == 1:
        choice = 1
    elif menu_count == 2:
        choice = 2
    else:
        choice = 4
    
    if choice == 1:
        print("  Option 1: View Profile")
    elif choice == 2:
        print("  Option 2: Edit Settings")
    elif choice == 3:
        print("  Option 3: Help")
    elif choice == 4:
        print("  Option 4: Exit")
    else:
        print("  Invalid choice!")

print("\n" + "="*60)
print("PRACTICAL EXAMPLES")
print("="*60)

# Example 1: Number guessing game simulation
print("\n1. NUMBER GUESSING GAME SIMULATION")
secret_number = 7
guess = 1
attempts = 0
max_attempts = 3

print(f"Guessing game (secret number is {secret_number}):")
while guess != secret_number and attempts < max_attempts:
    attempts += 1
    if attempts == 1:
        guess = 5
    elif attempts == 2:
        guess = 8
    else:
        guess = 7
    
    if guess < secret_number:
        print(f"  Attempt {attempts}: {guess} - Too low!")
    elif guess > secret_number:
        print(f"  Attempt {attempts}: {guess} - Too high!")
    else:
        print(f"  Attempt {attempts}: {guess} - Correct!")

# Example 2: Processing list until empty
print("\n2. PROCESSING LIST UNTIL EMPTY")
tasks = ['task1', 'task2', 'task3']
print("Processing tasks:")
while tasks:
    current_task = tasks.pop(0)
    print(f"  Processing: {current_task}")
    print(f"  Remaining tasks: {tasks}")

# Example 3: Fibonacci sequence
print("\n3. FIBONACCI SEQUENCE")
a, b = 0, 1
count = 0
print("First 8 Fibonacci numbers:")
while count < 8:
    print(f"  {a}")
    a, b = b, a + b
    count += 1

print("\n" + "="*60)
print("KEY DIFFERENCES: FOR vs WHILE LOOPS")
print("="*60)
print("FOR LOOPS:")
print("- Use when you know the number of iterations")
print("- Iterate over sequences (lists, strings, ranges)")
print("- Automatically handle iteration")
print("- More Pythonic for most cases")
print("- Use with: range(), lists, strings, dictionaries")
print()
print("WHILE LOOPS:")
print("- Use when you don't know exact number of iterations")
print("- Continue until a condition becomes False")
print("- Manual control over iteration")
print("- Risk of infinite loops if not careful")
print("- Use with: user input, file reading, condition-based logic")
print("="*60)