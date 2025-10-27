
# ============================================================
# PART 1: BASIC TRY-EXCEPT
# ============================================================

print("\n" + "="*60)
print("PART 1: BASIC TRY-EXCEPT")
print("="*60)

# Example 1: Basic exception handling
print("\n1. BASIC EXCEPTION HANDLING:")
try:
    result = 10 / 2
    print(f"Result: {result}")
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Example 2: Handling different exceptions
print("\n2. HANDLING DIFFERENT EXCEPTIONS:")
try:
    number = int("abc")  # This will raise ValueError
except ValueError as e:
    print(f"ValueError occurred: {e}")

# Example 3: Multiple exception types
print("\n3. MULTIPLE EXCEPTION TYPES:")
def divide_numbers(a, b):
    try:
        result = a / b
        print(f"{a} / {b} = {result}")
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
    except TypeError:
        print("Error: Invalid input type!")

divide_numbers(10, 2)  # Normal case
divide_numbers(10, 0)  # ZeroDivisionError
divide_numbers("10", 2)  # TypeError

# ============================================================
# PART 2: EXCEPT WITH ERROR MESSAGES
# ============================================================

print("\n" + "="*60)
print("PART 2: EXCEPT WITH ERROR MESSAGES")
print("="*60)

# Example 1: Catching and printing error message
print("\n1. CATCHING AND PRINTING ERROR MESSAGE:")
try:
    numbers = [1, 2, 3]
    print(numbers[5])  # IndexError
except IndexError as error:
    print(f"Error: {error}")
    print(f"Error type: {type(error).__name__}")

# Example 2: Getting error details
print("\n2. GETTING ERROR DETAILS:")
try:
    value = None
    print(value.upper())  # AttributeError
except AttributeError as e:
    print(f"AttributeError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

# Example 3: File handling
print("\n3. FILE HANDLING EXCEPTIONS:")
try:
    with open("nonexistent_file.txt", "r") as file:
        content = file.read()
except FileNotFoundError as e:
    print(f"File not found: {e}")

# ============================================================
# PART 3: ELSE CLAUSE
# ============================================================

print("\n" + "="*60)
print("PART 3: ELSE CLAUSE")
print("="*60)

# Example 1: Try-else-finally
print("\n1. TRY-ELSE-FINALLY:")
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print("Error: Division by zero hereeeee",e)
else:
    print(f"No error occurred! Result: {result}")
finally:
    print("Finally block always executes")

# Example 2: Another try-else example
print("\n2. ANOTHER TRY-ELSE EXAMPLE:")
try:
    number = int("123")
except ValueError:
    print("Invalid number!")
else:
    print(f"Valid number entered: {number}")
finally:
    print("Process completed")