#Lists are mutable starts with square bracket
#Tuples are immutable and starts with small bracket

items = [1,True, 5.5, 'satya'];
print("Original list:", items)

# U can append something at end of the list 
items.append('Tanima')
print("After appending 'Tanima':", items)

items[1] = 'satya'
print('after changing list ',items)

items.extend([1.2222,3.222,4.22222])
print('printing after extending list ',items)

#checking pop operation

x = items.pop()
print(x)

# Tuples are immutable so extend, append , wont work

y = (1,3,4,5,6,7)

print(y)


# Create sample lists for demonstration
numbers = [1, 2, 3, 4, 5]
fruits = ['apple', 'banana', 'orange', 'apple', 'grape']
mixed = [1, 'hello', 3.14, True, [1, 2, 3]]

print(f"Sample lists:")
print(f"numbers: {numbers}")
print(f"fruits: {fruits}")
print(f"mixed: {mixed}")

print("\n" + "="*60)
print("1. ADDING ELEMENTS")
print("="*60)

# append() - Add single element at the end
print("\n1.1 APPEND() - Add single element at end")
numbers.append(6)
print(f"numbers.append(6): {numbers}")

# extend() - Add multiple elements from another list
print("\n1.2 EXTEND() - Add multiple elements")
numbers.extend([7, 8, 9])
print(f"numbers.extend([7, 8, 9]): {numbers}")

# insert() - Insert element at specific index
print("\n1.3 INSERT() - Insert at specific position")
numbers.insert(0, 0)  # Insert 0 at index 0
print(f"numbers.insert(0, 0): {numbers}")
numbers.insert(3, 2.5)  # Insert 2.5 at index 3
print(f"numbers.insert(3, 2.5): {numbers}")

print("\n" + "="*60)
print("2. REMOVING ELEMENTS")
print("="*60)

# pop() - Remove and return element (default: last element)
print("\n2.1 POP() - Remove and return element")
last_item = numbers.pop()
print(f"numbers.pop(): removed {last_item}, list: {numbers}")
second_item = numbers.pop(1)  # Remove element at index 1
print(f"numbers.pop(1): removed {second_item}, list: {numbers}")

# remove() - Remove first occurrence of value
print("\n2.2 REMOVE() - Remove first occurrence of value")
fruits.remove('apple')  # Remove first 'apple'
print(f"fruits.remove('apple'): {fruits}")

# clear() - Remove all elements
print("\n2.3 CLEAR() - Remove all elements")
temp_list = [1, 2, 3]
print(f"Before clear: {temp_list}")
temp_list.clear()
print(f"After clear: {temp_list}")

print("\n" + "="*60)
print("3. SEARCHING AND COUNTING")
print("="*60)

# index() - Find index of first occurrence
print("\n3.1 INDEX() - Find index of element")
try:
    apple_index = fruits.index('banana')
    print(f"fruits.index('banana'): {apple_index}")
except ValueError:
    print("Element not found!")

# count() - Count occurrences of element
print("\n3.2 COUNT() - Count occurrences")
apple_count = fruits.count('apple')
print(f"fruits.count('apple'): {apple_count}")

# in operator - Check if element exists
print("\n3.3 IN OPERATOR - Check membership")
print(f"'banana' in fruits: {'banana' in fruits}")
print(f"'mango' in fruits: {'mango' in fruits}")

print("\n" + "="*60)
print("4. SORTING AND REVERSING")
print("="*60)

# sort() - Sort list in place
print("\n4.1 SORT() - Sort list in place")
numbers_to_sort = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"Before sort: {numbers_to_sort}")
numbers_to_sort.sort()
print(f"After sort: {numbers_to_sort}")

# sort() with reverse parameter
numbers_to_sort.sort(reverse=True)
print(f"After sort(reverse=True): {numbers_to_sort}")

# sorted() - Return new sorted list
print("\n4.2 SORTED() - Return new sorted list")
original = [3, 1, 4, 1, 5]
sorted_list = sorted(original)
print(f"Original: {original}")
print(f"Sorted: {sorted_list}")

# reverse() - Reverse list in place
print("\n4.3 REVERSE() - Reverse list in place")
numbers_to_reverse = [1, 2, 3, 4, 5]
print(f"Before reverse: {numbers_to_reverse}")
numbers_to_reverse.reverse()
print(f"After reverse: {numbers_to_reverse}")

print("\n" + "="*60)
print("5. COPYING AND SLICING")
print("="*60)

# copy() - Create shallow copy
print("\n5.1 COPY() - Create shallow copy")
original_list = [1, 2, 3, [4, 5]]
copied_list = original_list.copy()
print(f"Original: {original_list}")
print(f"Copied: {copied_list}")
print(f"Same object? {original_list is copied_list}")

# Slicing
print("\n5.2 SLICING - Extract parts of list")
sample = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"Original: {sample}")
print(f"First 3 elements: {sample[:3]}")
print(f"Last 3 elements: {sample[-3:]}")
print(f"Middle elements: {sample[2:7]}")
print(f"Every 2nd element: {sample[::2]}")
print(f"Reverse: {sample[::-1]}")

print("\n" + "="*60)
print("6. LIST COMPREHENSION")
print("="*60)

# List comprehension examples
print("\n6.1 BASIC LIST COMPREHENSION")
squares = [x**2 for x in range(1, 6)]
print(f"Squares: {squares}")

# With condition
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
print(f"Even squares: {even_squares}")

# String manipulation
words = ['hello', 'world', 'python']
upper_words = [word.upper() for word in words]
print(f"Upper words: {upper_words}")

print("\n" + "="*60)
print("7. PRACTICAL EXAMPLES")
print("="*60)

# Example 1: Shopping cart
print("\n7.1 SHOPPING CART EXAMPLE")
cart = []
cart.append('apple')
cart.append('banana')
cart.extend(['milk', 'bread'])
print(f"Shopping cart: {cart}")

# Remove item
cart.remove('banana')
print(f"After removing banana: {cart}")

# Example 2: Student grades
print("\n7.2 STUDENT GRADES EXAMPLE")
grades = [85, 92, 78, 96, 88]
print(f"Grades: {grades}")
print(f"Average: {sum(grades) / len(grades):.2f}")
print(f"Highest grade: {max(grades)}")
print(f"Lowest grade: {min(grades)}")

# Example 3: Unique elements
print("\n7.3 REMOVE DUPLICATES")
duplicates = [1, 2, 2, 3, 3, 3, 4, 5]
unique = list(set(duplicates))  # Convert to set to remove duplicates, then back to list
print(f"Original: {duplicates}")
print(f"Unique: {unique}")

print("\n" + "="*60)
print("8. COMMON LIST OPERATIONS")
print("="*60)

# Length
print(f"\n8.1 LEN() - Get length")
print(f"Length of numbers: {len(numbers)}")

# Sum, min, max
print(f"\n8.2 SUM, MIN, MAX")
numeric_list = [1, 2, 3, 4, 5]
print(f"Sum: {sum(numeric_list)}")
print(f"Min: {min(numeric_list)}")
print(f"Max: {max(numeric_list)}")

# Concatenation
print(f"\n8.3 CONCATENATION")
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2
print(f"Combined: {combined}")

# Repetition
print(f"\n8.4 REPETITION")
repeated = [1, 2] * 3
print(f"Repeated: {repeated}")

print("\n" + "="*60)
print("9. TUPLES - IMMUTABLE LISTS")
print("="*60)

# Tuple examples
print("\n9.1 TUPLE CREATION")
coordinates = (10, 20)
colors = ('red', 'green', 'blue')
single_tuple = (42,)  # Note the comma for single element
print(f"Coordinates: {coordinates}")
print(f"Colors: {colors}")
print(f"Single tuple: {single_tuple}")

# Tuple operations
print("\n9.2 TUPLE OPERATIONS")
print(f"Length: {len(coordinates)}")
print(f"Index 0: {coordinates[0]}")
print(f"Slicing: {coordinates[0:1]}")

# Tuple unpacking
print("\n9.3 TUPLE UNPACKING")
x, y = coordinates
print(f"Unpacked: x={x}, y={y}")

# Convert between list and tuple
print("\n9.4 CONVERSION")
list_from_tuple = list(coordinates)
tuple_from_list = tuple([1, 2, 3])
print(f"List from tuple: {list_from_tuple}")
print(f"Tuple from list: {tuple_from_list}")

print("\n" + "="*60)
print("KEY DIFFERENCES: LISTS vs TUPLES")
print("="*60)
print("LISTS:")
print("- Mutable (can be changed)")
print("- Use square brackets []")
print("- Methods: append(), extend(), remove(), pop(), etc.")
print("- Slower than tuples")
print("- Use when you need to modify the collection")
print()
print("TUPLES:")
print("- Immutable (cannot be changed)")
print("- Use parentheses ()")
print("- No modification methods")
print("- Faster than lists")
print("- Use when you need fixed, unchangeable data")
print("="*60)



