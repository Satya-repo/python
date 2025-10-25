#set is unique unordered collection

myset = {1,2,3,4,3,4,7,9}
print('initial set is ',myset)

myset.add(99)

print('new set is ',myset)

myset.remove(3)

print('new set is ',myset)

print(f"{'+' * 60}")

#Dictionary in action

# Creating a dictionary
mydict = {
    "name": "Satya",
    "age": 30,
    "city": "Bangalore"
}

mymovie = {
    "name": "It ends here",
    "actress": "Suunny",
    "year": "1998"
}

print("Dictionary:", mydict)
print("Dictionary : ",mymovie)

print("Actess ",mymovie["actress"])
print("Name ",mymovie["name"])

mymovie["country"] = "India"
print('reprinting dictionaty ',mymovie)


# # Updating value for an existing key
mydict["age"] = 37
print("After updating age:", mydict)

#Iterating dictionary 

for key,value in mymovie.items():
    print('key is ',key , ' n value is ',value)


# Sample dictionaries for demonstration
student_info = {
    "name": "Alice",
    "age": 20,
    "grade": "A",
    "subjects": ["Math", "Science", "English"],
    "gpa": 3.8
}


print(f"student_info: {student_info}")


# Method 1: Using .keys() method

for key in student_info.keys():
    print(f"  Key: {key}")

# Method 2: Direct iteration (default behavior)

for key in student_info:
    print(f"  Key: {key}")

# Method 3: Using .keys() with explicit iteration
print("\n1.3 Using .keys() with explicit iteration:")
for key in student_info.keys():
    value = student_info[key]
    print(f"  {key}: {value}")



# Using .values() method
print(" Using .values() method:")
for value in student_info.values():
    print(f"  Value: {value}")


# Method 1: Using .items() method (RECOMMENDED)
print("Using .items() method (RECOMMENDED):")
for key, value in student_info.items():
    print(f"  {key}: {value}")


# Method 3: Manual iteration with keys
print(" Manual iteration with keys:")
for key in student_info:
    value = student_info[key]
    print(f"  {key}: {value}")



# Iterate only over specific keys
print("\n4.1 Iterate only over specific keys:")
important_keys = ["name", "age", "gpa"]
for key in important_keys:
    if key in student_info:
        print(f"  {key}: {student_info[key]}")

