x = 23
y = 46
z = x*y
z1 = x**y #this is way of deffing exponent that is 23 base to power 46
print(z,z1)

a = 10
b = 3
c = a/b
c1 = a//b

print(c,c1)

c2 = a%b
print(c2)

d = (a*b)**4

print(d)

num = input('num is ') # input function return string as type
print(int(num)-5)

print("\n" + "="*60)
print("BODMAS EXAMPLES IN PYTHON")
print("="*60)

# BODMAS = Brackets, Orders, Division, Multiplication, Addition, Subtraction
# In Python: (), **, *, /, //, %, +, -

print("\n1. BRACKETS (Parentheses) - Highest Priority")
print("Example: 2 + 3 * 4 =", 2 + 3 * 4)  # 14 (multiplication first)
print("With brackets: (2 + 3) * 4 =", (2 + 3) * 4)  # 20 (brackets first)

print("\n2. ORDERS (Exponentiation) - **")
print("Example: 2 ** 3 + 4 =", 2 ** 3 + 4)  # 8 + 4 = 12
print("With brackets: 2 ** (3 + 4) =", 2 ** (3 + 4))  # 2^7 = 128

print("\n3. DIVISION AND MULTIPLICATION (Same priority, left to right)")
print("Example: 10 / 2 * 3 =", 10 / 2 * 3)  # 5 * 3 = 15
print("Example: 10 * 2 / 3 =", 10 * 2 / 3)  # 20 / 3 = 6.666...

print("\n4. ADDITION AND SUBTRACTION (Same priority, left to right)")
print("Example: 10 + 5 - 3 =", 10 + 5 - 3)  # 15 - 3 = 12
print("Example: 10 - 5 + 3 =", 10 - 5 + 3)  # 5 + 3 = 8

print("\n5. COMPLEX EXAMPLES SHOWING BODMAS:")
print("Example 1: 2 + 3 * 4 ** 2 =", 2 + 3 * 4 ** 2)
print("Step by step: 2 + 3 * 16 = 2 + 48 = 50")

print("\nExample 2: (2 + 3) * 4 ** 2 =", (2 + 3) * 4 ** 2)
print("Step by step: 5 * 16 = 80")

print("\nExample 3: 2 + 3 * 4 / 2 =", 2 + 3 * 4 / 2)
print("Step by step: 2 + 12 / 2 = 2 + 6 = 8")

print("\nExample 4: (2 + 3) * (4 + 5) =", (2 + 3) * (4 + 5))
print("Step by step: 5 * 9 = 45")

print("\nExample 5: 2 ** 3 + 4 * 5 - 6 / 2 =", 2 ** 3 + 4 * 5 - 6 / 2)
print("Step by step: 8 + 20 - 3 = 25")

print("\n6. FLOOR DIVISION AND MODULO:")
print("Example: 10 // 3 * 2 =", 10 // 3 * 2)  # 3 * 2 = 6
print("Example: 10 % 3 + 2 =", 10 % 3 + 2)    # 1 + 2 = 3

print("\n7. MIXED OPERATIONS:")
print("Example: 2 + 3 * 4 ** 2 // 5 =", 2 + 3 * 4 ** 2 // 5)
print("Step by step: 2 + 3 * 16 // 5 = 2 + 48 // 5 = 2 + 9 = 11")

print("\n8. COMPARISON WITH DIFFERENT GROUPINGS:")
expression1 = "2 + 3 * 4"
expression2 = "(2 + 3) * 4"
expression3 = "2 + (3 * 4)"

print(f"{expression1} = {eval(expression1)}")
print(f"{expression2} = {eval(expression2)}")
print(f"{expression3} = {eval(expression3)}")

print("\n9. PRACTICAL EXAMPLES:")
# Area calculation
length = 10
width = 5
height = 3
area = length * width + 2 * (length * height + width * height)
print(f"Surface area: {length} * {width} + 2 * ({length} * {height} + {width} * {height}) = {area}")

# Temperature conversion
celsius = 25
fahrenheit = celsius * 9 / 5 + 32
print(f"Temperature: {celsius}°C = {celsius} * 9 / 5 + 32 = {fahrenheit}°F")

print("\n" + "="*60)
print("REMEMBER: Use brackets to control order of operations!")
print("="*60)