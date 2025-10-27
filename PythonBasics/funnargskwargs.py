# ============================================================
# COMPREHENSIVE FUNCTION EXAMPLES: ARGS, KWARGS, AND MORE
# ============================================================

from ast import arg


print("="*60)
print("PYTHON FUNCTIONS: ARGS AND KWARGS COMPREHENSIVE GUIDE")
print("="*60)

# ============================================================
# PART 1: BASIC FUNCTIONS
# ============================================================

# Example 1: Simple function
def greet(name):
   
    return f"Hello, {name}!"


print(greet("Alice"))

# Example 2: Function with default parameters
def greet_default(name, greeting="Hello"):
  
    return f"{greeting}, {name}!"

print("\n2. FUNCTION WITH DEFAULT PARAMETERS:")
print(greet_default("Bob"))
print(greet_default("Bob", "Hi"))


# ============================================================
# PART 2: *ARGS (Variable Positional Arguments)
# ============================================================

print("\n" + "="*60)
print("PART 2: *ARGS - VARIABLE POSITIONAL ARGUMENTS")
print("="*60)

# Example 1: Basic *args
def sum_numbers(*args):
   
    print(f"args type: {type(args)}")
    print(f"args value: {args}")
    return sum(args)

result = sum_numbers(1, 2, 3)
print(f"sum_numbers(1, 2, 3) = {result}")

def max(*args):
    if type(args) == int:
        print('printing max ',max(args))
    


max(1,2,3,4,5,6,7,7)
result2 = sum_numbers(1, 2, 3, 4, 5, 10)
print(f"sum_numbers(1, 2, 3, 4, 5, 10) = {result2}")

# Example 2: *args with other parameters
def greet_many(greeting, *names):
    """Function with regular parameter and *args"""
    print(f"\n{greeting}!")
    for name in names:
        print(f"  Hello, {name}")

print("\n2. *ARGS WITH OTHER PARAMETERS:")
greet_many("Welcome", "Alice", "Bob", "Charlie")

# Example 3: Practical use case - calculate average
def calculate_average(*numbers):
    """Calculate average of any number of values"""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

print("\n3. PRACTICAL USE - CALCULATE AVERAGE:")
print(f"Average of (10, 20, 30): {calculate_average(10, 20, 30)}")
print(f"Average of (5, 10, 15, 20, 25): {calculate_average(5, 10, 15, 20, 25)}")
print(f"Average of (100, 200): {calculate_average(100, 200)}")

# Example 4: Finding max and min
def find_max_min(*args):
    """Find maximum and minimum from variable arguments"""
    if not args:
        return None, None
    return max(args), min(args)

print("\n4. FINDING MAX AND MIN WITH *ARGS:")
max_val, min_val = find_max_min(10, 5, 8, 3, 12, 7)
print(f"Values: 10, 5, 8, 3, 12, 7")
print(f"Max: {max_val}, Min: {min_val}")

# ============================================================
# PART 3: **KWARGS (Variable Keyword Arguments)
# ============================================================

print("\n" + "="*60)
print("PART 3: **KWARGS - VARIABLE KEYWORD ARGUMENTS")
print("="*60)

# Example 1: Basic **kwargs
def print_info(**kwargs):
    """Accept any number of keyword arguments"""
    print(f"kwargs type: {type(kwargs)}")
    print(f"kwargs value: {kwargs}")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

print("\n1. BASIC **KWARGS:")
print_info(name="Alice", age=25, city="New York")

print_info(name="Bob", age=30, city="London", occupation="Engineer")

# Example 2: **kwargs with default values
def create_profile(**kwargs):
    """Create profile with optional parameters"""
    profile = {
        "name": kwargs.get("name", "Unknown"),
        "age": kwargs.get("age", 0),
        "city": kwargs.get("city", "Unknown"),
        "email": kwargs.get("email", "No email provided")
    }
    return profile

print("\n2. **KWARGS WITH DEFAULT VALUES:")
profile1 = create_profile(name="Alice", age=25)
print(profile1)

profile2 = create_profile(name="Bob", age=30, city="Tokyo", email="bob@example.com")
print(profile2)

profile3 = create_profile()  # No arguments
print(profile3)

# Example 3: Practical use case - Student record
def create_student(name, **info):
    """Create student record with flexible info"""
    student = {"name": name}
    student.update(info)  # Add all extra info
    return student

print("\n3. PRACTICAL USE - STUDENT RECORD:")
student1 = create_student("Alice", age=20, grade="A", gpa=3.8)
print(f"Student 1: {student1}")

student2 = create_student("Bob", age=22, grade="B", gpa=3.5, major="CS")
print(f"Student 2: {student2}")

# ============================================================
# PART 4: COMBINING *ARGS AND **KWARGS
# ============================================================

print("\n" + "="*60)
print("PART 4: COMBINING *ARGS AND **KWARGS")
print("="*60)

# Example 1: Basic combination
def flexible_function(*args, **kwargs):
    """Function accepting both *args and **kwargs"""
    print("Positional arguments:")
    for arg in args:
        print(f"  - {arg}")
    
    print("Keyword arguments:")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

print("\n1. COMBINING *ARGS AND **KWARGS:")
flexible_function(1, 2, 3, name="Alice", age=25, city="NYC")

# Example 2: Order matters!
def process_data(required_param, *args, **kwargs):
    """Function with required, *args, and **kwargs"""
    print(f"Required: {required_param}")
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

print("\n2. WITH REQUIRED PARAMETER:")
process_data("Hello", "arg1", "arg2", key1="value1", key2="value2")

# Example 3: Creating flexible print function
def custom_print(*args, sep=" ", end="\n", file=None):
    """Custom print function using *args and **kwargs"""
    output = sep.join(str(arg) for arg in args)
    if file is None:
        print(output, end=end)
    else:
        file.write(output + end)

print("\n3. CUSTOM PRINT FUNCTION:")
custom_print("Custom", "print", "function", sep="-", end="!\n")

# Example 4: Universal function wrapper
def logger(func, *args, **kwargs):
    """Log function calls with arguments"""
    print(f"\nCalling function: {func.__name__}")
    print(f"Positional args: {args}")
    print(f"Keyword args: {kwargs}")
    result = func(*args, **kwargs)
    print(f"Result: {result}")
    return result

def add(a, b):
    return a + b

print("\n4. FUNCTION LOGGER:")
logger(add, 5, 3)
logger(add, a=10, b=20)
