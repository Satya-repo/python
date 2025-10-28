import numpy as np
# ============================================================
# PART 1: CREATING NUMPY ARRAYS
# ============================================================

print("="*60)
print("PART 1: CREATING NUMPY ARRAYS")
print("="*60)

# Example 1: Creating arrays from lists
print("\n1. CREATING ARRAYS FROM LISTS:")
list_data = [1, 2, 3, 4, 5]
arr = np.array(list_data)
arr1 = np.array(range(1,10000,2))
print('printing range array ',arr1)
print(f"List: {list_data}")
print(f"NumPy array: {arr}")
print(f"Array shape: {arr.shape}")

# Example 2: Creating multidimensional arrays
print("\n2. MULTIDIMENSIONAL ARRAYS:")
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Matrix:\n{matrix}")
print(f"Shape: {matrix.shape}")
print(f"Dimensions: {matrix.ndim}")

# Example 3: Creating arrays with zeros and ones
print("\n3. ARRAYS WITH ZEROS AND ONES:")
zeros = np.zeros((3, 4))
ones = np.ones((2, 5))
myarray = np.ones((15,15))
print(f"Zeros array (3x4):\n{zeros}")
print(f"\nOnes array (2x5):\n{ones}")
print('printing my aray ',myarray)


# Example 7: Random arrays
print("\n7. RANDOM ARRAYS:")
random_arr = np.random.rand(3, 3)
random_int = np.random.randint(1, 10, size=(3, 3))
print(f"Random array (1-2):\n{random_arr}")
print(f"\nRandom integers (1-10):\n{random_int}")

# ============================================================
# PART 2: ARRAY PROPERTIES AND ATTRIBUTES
# ============================================================

print("\n" + "="*60)
print("PART 2: ARRAY PROPERTIES AND ATTRIBUTES")
print("="*60)

# Example 1: Array attributes
print("\n1. ARRAY ATTRIBUTES:")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Array:\n{arr}")
print(f"Shape: {arr.shape}")
print(f"Size: {arr.size}")
print(f"Dimensions: {arr.ndim}")
print(f"Data type: {arr.dtype}")
print(f"Item size: {arr.itemsize} bytes")
print(f"Total bytes: {arr.nbytes}")

# Example 2: Array types
print("\n2. ARRAY DATA TYPES:")
int_arr = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1.5, 2.7, 3.1], dtype=np.float64)
print(f"Integer array: {int_arr} (type: {int_arr.dtype})")
print(f"Float array: {float_arr} (type: {float_arr.dtype})")

# ============================================================
# PART 3: ARRAY INDEXING AND SLICING
# ============================================================

print("\n" + "="*60)
print("PART 3: ARRAY INDEXING AND SLICING")
print("="*60)

# Example 1: Basic indexing
print("\n1. BASIC INDEXING:")
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"First element: {arr[0]}")
print(f"Last element: {arr[-1]}")
print(f"Index 2: {arr[2]}")

# Example 2: Slicing
print("\n2. SLICING:")
print(f"First 3 elements: {arr[:3]}")
print(f"Last 3 elements: {arr[-3:]}")
print(f"Elements 2 to 4: {arr[1:4]}")
print(f"Every other element: {arr[::2]}")

# Example 3: Multidimensional indexing
print("\n3. MULTIDIMENSIONAL INDEXING:")
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Matrix:\n{matrix}")
print(f"Element at [1, 2]: {matrix[1, 2]}")
print(f"First row: {matrix[0, :]}")
print(f"First column: {matrix[:, 0]}")
print(f"Submatrix:\n{matrix[1:, 1:]}")

# Example 4: Boolean indexing
print("\n4. BOOLEAN INDEXING:")
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 2
filtered = arr[mask]
print(f"Array: {arr}")
print(f"Mask (arr > 2): {mask}")
print(f"Filtered: {filtered}")

# ============================================================
# PART 4: ARRAY OPERATIONS
# ============================================================

print("\n" + "="*60)
print("PART 4: ARRAY OPERATIONS")
print("="*60)

# Example 1: Arithmetic operations
print("\n1. ARITHMETIC OPERATIONS:")
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])

print(f"Array 1: {arr1}")
print(f"Array 2: {arr2}")
print(f"Addition: {arr1 + arr2}")
print(f"Subtraction: {arr2 - arr1}")
print(f"Multiplication: {arr1 * arr2}")
print(f"Division: {arr2 / arr1}")
print(f"Power: {arr1 ** 2}")

# Example 2: Matrix operations
print("\n2. MATRIX OPERATIONS:")
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

print(f"Matrix 1:\n{matrix1}")
print(f"Matrix 2:\n{matrix2}")
print(f"\nElement-wise multiplication:\n{matrix1 * matrix2}")
print(f"\nMatrix multiplication:\n{np.dot(matrix1, matrix2)}")
print(f"\nUsing @ operator:\n{matrix1 @ matrix2}")

# Example 3: Statistical operations
print("\n3. STATISTICAL OPERATIONS:")
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Array: {arr}")
print(f"Sum: {np.sum(arr)}")
print(f"Mean: {np.mean(arr)}")
print(f"Std: {np.std(arr):.2f}")
print(f"Min: {np.min(arr)}")
print(f"Max: {np.max(arr)}")
print(f"Median: {np.median(arr)}")

# Example 4: Aggregate functions
print("\n4. AGGREGATE FUNCTIONS:")
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Matrix:\n{matrix}")
print(f"\nSum along axis 0 (columns): {matrix.sum(axis=0)}")
print(f"Sum along axis 1 (rows): {matrix.sum(axis=1)}")
print(f"Mean along axis 0: {matrix.mean(axis=0)}")
print(f"Mean along axis 1: {matrix.mean(axis=1)}")

# ============================================================
# PART 5: ARRAY MANIPULATION
# ============================================================

print("\n" + "="*60)
print("PART 5: ARRAY MANIPULATION")
print("="*60)

# Example 1: Reshape
print("\n1. RESHAPE:")
arr = np.array([1, 2, 3, 4, 5, 6])
print(f"Original: {arr}")
reshaped = arr.reshape(2, 3)
print(f"Reshaped (2x3):\n{reshaped}")

# Example 2: Flatten and ravel
print("\n2. FLATTEN AND RAVEL:")
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Matrix:\n{matrix}")
print(f"Flatten: {matrix.flatten()}")
print(f"Ravel: {matrix.ravel()}")

# Example 3: Transpose
print("\n3. TRANSPOSE:")
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Matrix:\n{matrix}")
print(f"Transposed:\n{matrix.T}")

# Example 4: Concatenate
print("\n4. CONCATENATE:")
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concatenated = np.concatenate([arr1, arr2])
print(f"Array 1: {arr1}")
print(f"Array 2: {arr2}")
print(f"Concatenated: {concatenated}")

# Example 5: Stack
print("\n5. STACK:")
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
vstack = np.vstack([arr1, arr2])
hstack = np.hstack([arr1, arr2])
print(f"Vertical stack:\n{vstack}")
print(f"Horizontal stack: {hstack}")

# ============================================================
# PART 6: ADVANCED OPERATIONS
# ============================================================

print("\n" + "="*60)
print("PART 6: ADVANCED OPERATIONS")
print("="*60)

# Example 1: Broadcasting
print("\n1. BROADCASTING:")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
scalar = 10
print(f"Array:\n{arr}")
print(f"Add scalar {scalar}: \n{arr + scalar}")
print(f"Multiply by 2: \n{arr * 2}")

# Example 2: Where function
print("\n2. WHERE FUNCTION:")
arr = np.array([1, 2, 3, 4, 5, 6])
result = np.where(arr > 3, arr, -1)
print(f"Array: {arr}")
print(f"Where arr > 3: {result}")

# Example 3: Vectorized operations
print("\n3. VECTORIZED OPERATIONS:")
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Square root: {np.sqrt(arr)}")
print(f"Exponential: {np.exp(arr)}")
print(f"Logarithm: {np.log(arr + 1)}")
print(f"Sine: {np.sin(arr)}")

# Example 4: Sorting
print("\n4. SORTING:")
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"Original: {arr}")
print(f"Sorted: {np.sort(arr)}")
print(f"Argsort (indices): {np.argsort(arr)}")

# Example 5: Unique values
print("\n5. UNIQUE VALUES:")
arr = np.array([1, 2, 2, 3, 3, 3, 4, 5])
print(f"Array: {arr}")
print(f"Unique values: {np.unique(arr)}")
print(f"Count of unique: {len(np.unique(arr))}")

# ============================================================
# PART 7: PRACTICAL EXAMPLES
# ============================================================

print("\n" + "="*60)
print("PART 7: PRACTICAL EXAMPLES")
print("="*60)

# Example 1: Image manipulation simulation
print("\n1. IMAGE MANIPULATION SIMULATION:")
# Simulate grayscale image (8x8 pixels, values 0-255)
image = np.random.randint(0, 256, (8, 8))
print(f"Original image (8x8):\n{image}")
normalized = image / 255.0
print(f"\nNormalized (0-1):\n{normalized}")
brightened = np.clip(image + 50, 0, 255)
print(f"\nBrightened (+50):\n{brightened}")

# Example 2: Data analysis
print("\n2. DATA ANALYSIS:")
scores = np.array([85, 92, 78, 96, 88, 95, 90, 87, 89, 91])
print(f"Test scores: {scores}")
print(f"Average: {np.mean(scores):.2f}")
print(f"Standard deviation: {np.std(scores):.2f}")
print(f"Highest score: {np.max(scores)}")
print(f"Lowest score: {np.min(scores)}")
print(f"Median: {np.median(scores)}")

# Example 3: Matrix operations
print("\n3. MATRIX OPERATIONS:")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")
print(f"\nA + B:\n{A + B}")
print(f"\nA * B (element-wise):\n{A * B}")
print(f"\nA @ B (matrix multiplication):\n{A @ B}")
print(f"\nDeterminant of A: {np.linalg.det(A):.2f}")

# Example 4: Linear algebra
print("\n4. LINEAR ALGEBRA:")
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
print(f"Matrix A:\n{A}")
print(f"Vector b: {b}")
x = np.linalg.solve(A, b)
print(f"Solution x (Ax=b): {x}")

# Example 5: Random data generation
print("\n5. RANDOM DATA GENERATION:")
normal_data = np.random.normal(0, 1, 10)
uniform_data = np.random.uniform(0, 10, 10)
print(f"Normal distribution (μ=0, σ=1): {normal_data}")
print(f"Uniform distribution (0-10): {uniform_data}")

# ============================================================
# PART 8: NUMERICAL COMPUTING
# ============================================================

print("\n" + "="*60)
print("PART 8: NUMERICAL COMPUTING")
print("="*60)

# Example 1: Efficient computations
print("\n1. EFFICIENT COMPUTATIONS:")
large_arr = np.random.rand(1000)
print(f"Array size: {large_arr.size}")
print(f"Sum: {np.sum(large_arr):.4f}")
print(f"Mean: {np.mean(large_arr):.4f}")
print(f"Standard deviation: {np.std(large_arr):.4f}")

# Example 2: Element-wise operations
print("\n2. ELEMENT-WISE OPERATIONS:")
x = np.array([1, 2, 3, 4, 5])
y = np.array([6, 7, 8, 9, 10])
print(f"x: {x}")
print(f"y: {y}")
print(f"x + y: {x + y}")
print(f"x * y: {x * y}")
print(f"x ** y: {x ** y}")

# Example 3: Masking and filtering
print("\n3. MASKING AND FILTERING:")
data = np.array([1, 5, 8, 12, 15, 18, 20, 25, 30, 35])
mask = (data >= 10) & (data <= 25)
filtered = data[mask]
print(f"Data: {data}")
print(f"Filtered (10-25): {filtered}")

# Example 4: Mathematical functions
print("\n4. MATHEMATICAL FUNCTIONS:")
angles = np.array([0, 30, 45, 60, 90])
radians = np.deg2rad(angles)
print(f"Angles (degrees): {angles}")
print(f"Sin: {np.sin(radians)}")
print(f"Cos: {np.cos(radians)}")
print(f"Tan: {np.tan(radians)}")

# ============================================================
# PART 9: QUICK REFERENCE
# ============================================================

print("\n" + "="*60)
print("PART 9: QUICK REFERENCE")
print("="*60)

print("""
ARRAY CREATION:
- np.array([1, 2, 3])
- np.zeros((3, 4))
- np.ones((2, 5))
- np.full((3, 3), 7)
- np.arange(0, 10, 2)
- np.linspace(0, 10, 5)
- np.random.rand(3, 3)

ARRAY PROPERTIES:
- arr.shape: dimensions
- arr.size: total elements
- arr.ndim: number of dimensions
- arr.dtype: data type
- arr.itemsize: bytes per element

COMMON OPERATIONS:
- arr + scalar: add scalar
- arr * 2: multiply by 2
- arr.T: transpose
- arr.reshape(rows, cols): reshape
- np.sum(arr): sum elements
- np.mean(arr): average
- np.std(arr): standard deviation

INDEXING:
- arr[0]: first element
- arr[-1]: last element
- arr[1:4]: slice
- arr > 3: boolean array
- arr[arr > 3]: filtering

MATHEMATICAL FUNCTIONS:
- np.sin(arr): sine
- np.cos(arr): cosine
- np.sqrt(arr): square root
- np.exp(arr): exponential
- np.log(arr): logarithm
""")

print("="*60)
print("END OF NUMPY EXAMPLES")
print("="*60)