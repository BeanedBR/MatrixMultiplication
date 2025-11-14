How to run it on Linux
-------------------------------------------------

1. Save the code (MatrixMultiplication.cpp)

2. Compile with g++ or clang++. For example:
g++ -std=c++17 -O2 MatrixMultiplication.cpp -pthread -o matmul

3. Run it:
./matmul

-------------------------------------------------

You'll get the same prompts:

Enter matrix size n (e.g. 300, 500, 700):
Enter number of threads (0 = use hardware_concurrency):
Show thread processes? (y/n):

-------------------------------------------------

Small things to be aware of:

- std::thread::hardware_concurrency( might return a different number, depending on your Linux machine's cores. That's fine.
- Performance may be a bit different due to compiler and CPU differences, but the behaviour and logic are the same.
