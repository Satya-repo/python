# ============================================================
# LAMBDA FUNCTIONS, MAP, AND FILTER EXAMPLES IN PYTHON
# ============================================================


# Example 1: Basic lambda functions
print("1. BASIC LAMBDA FUNCTIONS:")

# Regular function vs Lambda
def add_regular(x, y):
    return x + y

add_lambda = lambda x, y: x + y
sub_lambda = lambda x,y: x-y
print(f" 'checking subraction lamdba ', {sub_lambda(10,9)}")

print(f"Regular function: {add_regular(5, 3)}")
print(f"Lambda function: {add_lambda(5, 3)}")

# Example 2: Lambda with single parameter
print("2. LAMBDA WITH SINGLE PARAMETER:")
square = lambda x: x ** 2
cube = lambda x: x ** 3

print(f"Square of 5: {square(5)}")
print(f"Cube of 4: {cube(4)}")

# Example 3: Lambda with conditional logic
print("3. LAMBDA WITH CONDITIONAL LOGIC:")
is_even = lambda x: True if x % 2 == 0 else False
greater = lambda x, y: x if x > y else y
print(f"Is 6 even? {is_even(6)}")
print(f"Greater of 5 and 10: {greater(5, 10)}")



# ============================================================
# PART 2: MAP FUNCTION
# ============================================================

print("\n" + "="*60)
print("PART 2: MAP FUNCTION")
print("="*60)

# Example 1: Basic map with lambda
print("1. BASIC MAP WITH LAMBDA:")
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(f"Original: {numbers}")
print(f"Squared: {squared}")

# Example 2: Map with regular function
print("\n2. MAP WITH REGULAR FUNCTION:")
def double(x):
    return x * 2

doubled = list(map(double, numbers))
print(f"Doubled: {doubled}")

# Example 3: Map with multiple iterables
print("\n3. MAP WITH MULTIPLE ITERABLES:")
list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]
added = list(map(lambda x, y: x + y, list1, list2))
print(f"List1: {list1}")
print(f"List2: {list2}")
print(f"Added: {added}")

# Example 4: Map with strings
print("\n4. MAP WITH STRINGS:")
words = ["hello", "world", "python", "programming"]
lengths = list(map(lambda x: len(x), words))
uppercases = list(map(lambda x: x.upper(), words))
print(f"Words: {words}")
print(f"Lengths: {lengths}")
print(f"Uppercase: {uppercases}")

# ============================================================
# PART 3: FILTER FUNCTION
# ============================================================

print("\n" + "="*60)
print("PART 3: FILTER FUNCTION")
print("="*60)

# Example 1: Basic filter with lambda
print("\n1. BASIC FILTER WITH LAMBDA:")
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
odd_numbers = list(filter(lambda x: x % 2 != 0, numbers))
print(f"All numbers: {numbers}")
print(f"Even numbers: {even_numbers}")
print(f"Odd numbers: {odd_numbers}")

# Example 2: Filter with conditions
print("\n2. FILTER WITH CONDITIONS:")
scores = [45, 78, 89, 92, 65, 88, 95, 67, 85]
passing_scores = list(filter(lambda x: x >= 70, scores))
high_scores = list(filter(lambda x: x >= 90, scores))
print(f"All scores: {scores}")
print(f"Passing scores (>=70): {passing_scores}")
print(f"High scores (>=90): {high_scores}")

# Example 3: Filter with strings
print("\n3. FILTER WITH STRINGS:")
words = ["apple", "banana", "cherry", "date", "elderberry"]
short_words = list(filter(lambda x: len(x) <= 5, words))
long_words = list(filter(lambda x: len(x) > 5, words))
print(f"All words: {words}")
print(f"Short words (<=5 chars): {short_words}")
print(f"Long words (>5 chars): {long_words}")

# Example 4: Filter with None
print("\n4. FILTER WITH NONE:")
numbers = [1, None, 3, None, 5, None, 7]
without_none = list(filter(lambda x: x is not None, numbers))
print(f"Original: {numbers}")
print(f"Without None: {without_none}")

# Example 5: Filter with multiple conditions
print("\n5. FILTER WITH MULTIPLE CONDITIONS:")
ages = [12, 16, 18, 20, 25, 30, 35, 17]
adults = list(filter(lambda x: x >= 18, ages))
between = list(filter(lambda x: 18 <= x <= 30, ages))
print(f"All ages: {ages}")
print(f"Adults (>=18): {adults}")
print(f"Between 18-30: {between}")

# Example 6: Filter to extract specific data
print("\n6. FILTER TO EXTRACT SPECIFIC DATA:")
products = [
    {"name": "Laptop", "price": 999, "stock": 5},
    {"name": "Mouse", "price": 25, "stock": 0},
    {"name": "Keyboard", "price": 75, "stock": 12},
    {"name": "Monitor", "price": 299, "stock": 0}
]
in_stock = list(filter(lambda p: p["stock"] > 0, products))
expensive = list(filter(lambda p: p["price"] > 100, products))
print(f"All products: {products}")
print(f"In stock: {in_stock}")
print(f"Expensive (>$100): {expensive}")

# ============================================================
# PART 4: COMBINING LAMBDA, MAP, AND FILTER
# ============================================================

print("\n" + "="*60)
print("PART 4: COMBINING LAMBDA, MAP, AND FILTER")
print("="*60)

# Example 1: Chain map and filter
print("\n1. CHAIN MAP AND FILTER:")
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# First filter even numbers, then square them
even_squared = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))
print(f"Numbers: {numbers}")
print(f"Even numbers squared: {even_squared}")

# Example 2: Multiple operations
print("\n2. MULTIPLE OPERATIONS:")
numbers = [10, 20, 30, 40, 50, 60]
# Filter numbers > 25, then multiply by 2
result = list(map(lambda x: x * 2, filter(lambda x: x > 25, numbers)))
print(f"Numbers: {numbers}")
print(f"Numbers > 25 multiplied by 2: {result}")

# Example 3: Complex transformation
print("\n3. COMPLEX TRANSFORMATION:")
scores = [45, 78, 89, 92, 65, 88, 95, 67, 85]
# Filter passing scores, then convert to grades
grades = list(map(lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C', 
                  filter(lambda x: x >= 70, scores)))
print(f"Scores: {scores}")
print(f"Passing grades: {grades}")

# Example 4: Practical example - processing data
print("\n4. PRACTICAL EXAMPLE - PROCESSING DATA:")
employees = [
    {"name": "Alice", "age": 30, "salary": 50000},
    {"name": "Bob", "age": 25, "salary": 45000},
    {"name": "Charlie", "age": 35, "salary": 60000},
    {"name": "Diana", "age": 28, "salary": 48000}
]

# Get names of employees with salary > 47000
high_earners = list(map(lambda e: e["name"], 
                        filter(lambda e: e["salary"] > 47000, employees)))
print(f"Employees: {employees}")
print(f"High earners: {high_earners}")

# ============================================================
# PART 5: LAMBDA WITH BUILT-IN FUNCTIONS
# ============================================================

print("\n" + "="*60)
print("PART 5: LAMBDA WITH BUILT-IN FUNCTIONS")
print("="*60)

# Example 1: Lambda with sorted()
print("\n1. LAMBDA WITH SORTED():")
people = [
    ("Alice", 30),
    ("Bob", 25),
    ("Charlie", 35),
    ("Diana", 28)
]
sorted_by_age = sorted(people, key=lambda x: x[1])
sorted_by_name = sorted(people, key=lambda x: x[0])
print(f"People: {people}")
print(f"Sorted by age: {sorted_by_age}")
print(f"Sorted by name: {sorted_by_name}")

# Example 2: Lambda with max() and min()
print("\n2. LAMBDA WITH MAX() AND MIN():")
words = ["apple", "banana", "cherry", "date"]
longest = max(words, key=lambda x: len(x))
shortest = min(words, key=lambda x: len(x))
print(f"Words: {words}")
print(f"Longest: {longest}")
print(f"Shortest: {shortest}")

# Example 3: Lambda with sorted() with reverse
print("\n3. LAMBDA WITH SORTED() REVERSE:")
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_desc = sorted(numbers, key=lambda x: x, reverse=True)
print(f"Numbers: {numbers}")
print(f"Sorted descending: {sorted_desc}")

# ============================================================
# PART 6: PRACTICAL REAL-WORLD EXAMPLES
# ============================================================

print("\n" + "="*60)
print("PART 6: PRACTICAL REAL-WORLD EXAMPLES")
print("="*60)

# Example 1: Data processing pipeline
print("\n1. DATA PROCESSING PIPELINE:")
transactions = [
    {"product": "Laptop", "amount": 999, "discount": 10},
    {"product": "Mouse", "amount": 25, "discount": 5},
    {"product": "Keyboard", "amount": 75, "discount": 0},
    {"product": "Monitor", "amount": 299, "discount": 15}
]

# Calculate final prices and filter expensive items
final_prices = list(map(lambda t: {
    "product": t["product"],
    "final_price": t["amount"] * (1 - t["discount"] / 100)
}, transactions))

expensive_items = list(filter(lambda t: t["final_price"] > 200, final_prices))

print(f"Transactions: {transactions}")
print(f"Final prices: {final_prices}")
print(f"Expensive items (>$200): {expensive_items}")

# Example 2: Text processing
print("\n2. TEXT PROCESSING:")
sentences = ["Python is awesome", "I love programming", "Python is fun"]

# Filter sentences containing "Python" and convert to uppercase
filtered = list(map(lambda s: s.upper(), 
                   filter(lambda s: "Python" in s, sentences)))
print(f"Sentences: {sentences}")
print(f"Filtered and uppercased: {filtered}")

# Example 3: Number analysis
print("\n3. NUMBER ANALYSIS:")
numbers = [15, 20, 35, 40, 55, 60, 75, 80, 95, 100]

# Filter numbers divisible by 5 and 7, then square them
result = list(map(lambda x: x ** 2, 
                 filter(lambda x: x % 5 == 0 and x % 7 == 0, numbers)))
print(f"Numbers: {numbers}")
print(f"Numbers divisible by 5 and 7 (squared): {result}")

# ============================================================
# PART 7: LIST COMPREHENSIONS vs LAMBDA/MAP/FILTER
# ============================================================

print("\n" + "="*60)
print("PART 7: LIST COMPREHENSIONS vs LAMBDA/MAP/FILTER")
print("="*60)

# Example 1: Map vs List Comprehension
print("\n1. MAP vs LIST COMPREHENSION:")
numbers = [1, 2, 3, 4, 5]

# Using map
squared_map = list(map(lambda x: x ** 2, numbers))

# Using list comprehension
squared_lc = [x ** 2 for x in numbers]

print(f"Original: {numbers}")
print(f"Using map: {squared_map}")
print(f"Using list comprehension: {squared_lc}")

# Example 2: Filter vs List Comprehension
print("\n2. FILTER vs LIST COMPREHENSION:")
# Using filter
even_filter = list(filter(lambda x: x % 2 == 0, numbers))

# Using list comprehension
even_lc = [x for x in numbers if x % 2 == 0]

print(f"Using filter: {even_filter}")
print(f"Using list comprehension: {even_lc}")

# Example 3: Combined
print("\n3. COMBINED:")
# Using map and filter
even_squared_mf = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))

# Using list comprehension
even_squared_lc = [x ** 2 for x in numbers if x % 2 == 0]

print(f"Using map and filter: {even_squared_mf}")
print(f"Using list comprehension: {even_squared_lc}")

print("\n💡 TIP: List comprehensions are more Pythonic and often faster!")

# ============================================================
# PART 8: QUICK REFERENCE
# ============================================================

print("\n" + "="*60)
print("PART 8: QUICK REFERENCE")
print("="*60)

print("""
LAMBDA SYNTAX:
lambda arguments: expression

Examples:
- lambda x: x * 2
- lambda x, y: x + y
- lambda x: x ** 2 if x > 0 else 0

MAP SYNTAX:
map(function, iterable)
map(lambda x: x * 2, [1, 2, 3])

FILTER SYNTAX:
filter(function, iterable)
filter(lambda x: x > 5, [1, 3, 5, 7, 9])

WHEN TO USE:
✅ Lambda: Small, one-line functions
✅ Map: Transform each element
✅ Filter: Select elements based on condition
❌ Complex logic: Use regular functions instead

ALTERNATIVES:
- List comprehensions are often more readable
- [x**2 for x in range(10)] vs map(lambda x: x**2, range(10))
- [x for x in range(10) if x%2==0] vs filter(lambda x: x%2==0, range(10))
""")

print("="*60)
print("END OF LAMBDA, MAP, AND FILTER EXAMPLES")
print("="*60)

