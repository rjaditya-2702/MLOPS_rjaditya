# GitHub_Labs: Lab1

This lab focuses on 5 modules, which includes creating a virtual environment, creating a GitHub repository, creating Python files, creating test files using pytest and unittest, and implementing GitHub Actions.
Source: [MLOps/Labs/GitHub_Labs/Lab1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab1)

## LAB WORK 

### STEP1: Creating Virtual Environment
(i use anaconda)
```
conda create -n MLOps python=3.10.18
```

After creation, activate the environment:
```
conda activate MLOps
```
_Note: If conda activate doesn't work, you need to find the path where your conda is installe and use that to activate your environment (refer to terminal output)_

**Terminal output**
```
rjaditya@Adityas-MacBook-Air MLOps_rjaditya % source /opt/homebrew/anaconda3/bin/activate MLOps
(MLOps) rjaditya@Adityas-MacBook-Air MLOps_rjaditya % 
```

### STEP2: Creating a GitHub Repository, Cloning and Folder Structure

- Fork of Course Repo: https://github.com/rjaditya-2702/MLOps
- Cloning Course Repo to local terminal: 
```
rjaditya@Adityas-MacBook-Air MLOps % pwd
/Users/rjaditya/Documents/NEU-SEM/Fall-25/MLOps
rjaditya@Adityas-MacBook-Air MLOps % ls -l
total 0
drwxr-xr-x@ 9 rjaditya  staff  288 Oct 19 17:03 MLOps_project
drwxr-xr-x@ 8 rjaditya  staff  256 Oct 20 12:58 MLOps_rjaditya
rjaditya@Adityas-MacBook-Air MLOps % git clone https://github.com/raminmohammadi/MLOps.git
Cloning into 'MLOps'...
remote: Enumerating objects: 5234, done.
remote: Counting objects: 100% (39/39), done.
remote: Compressing objects: 100% (17/17), done.
remote: Total 5234 (delta 30), reused 22 (delta 22), pack-reused 5195 (from 3)
Receiving objects: 100% (5234/5234), 328.84 MiB | 13.59 MiB/s, done.
Resolving deltas: 100% (2558/2558), done.
Updating files: 100% (613/613), done.
rjaditya@Adityas-MacBook-Air MLOps % 
```
- Creating my own repository: https://github.com/rjaditya-2702/MLOPS_rjaditya
> I use this repository to implement Lab work. Each folder at the root corresponds to one lab submission in canvas
> The commit history in this repo captures the `git add .`,  `git commit`, `git pull`, and `git push` information

### STEP3: Creating calculator.py
```
from src.calculator import *

# to add
fun1(x,y)

# to substract b from a i.e. a - b
fun2(b,a)

# to multiply
fun3(a,b)

# to get sum of all above operations
fun4(a,b)
```
### STEP4: Testing
**pytest**
`pytest test/pytest_sample.py`
```
(MLOps) rjaditya@Adityas-MacBook-Air LAB3 % pytest test/pytest_sample.py -v
================================================================================================================= test session starts =================================================================================================================
platform darwin -- Python 3.10.18, pytest-8.4.2, pluggy-1.6.0 -- /opt/homebrew/anaconda3/envs/MLOps/bin/python3.10
cachedir: .pytest_cache
rootdir: /Users/rjaditya/Documents/NEU-SEM/Fall-25/MLOps/MLOps_rjaditya/LAB3
plugins: Faker-37.11.0, anyio-3.7.1
collected 24 items                                                                                                                                                                                                                                    

test/pytest_sample.py::test_fun1_addition[2-3-5] PASSED                                                                                                                                                                                         [  4%]
test/pytest_sample.py::test_fun1_addition[-2-3-1] PASSED                                                                                                                                                                                        [  8%]
test/pytest_sample.py::test_fun1_addition[3--2-1] PASSED                                                                                                                                                                                        [ 12%]
test/pytest_sample.py::test_fun1_addition[-3--5--8] PASSED                                                                                                                                                                                      [ 16%]
test/pytest_sample.py::test_fun1_addition[0-0-0] PASSED                                                                                                                                                                                         [ 20%]
test/pytest_sample.py::test_fun1_invalid_types PASSED                                                                                                                                                                                           [ 25%]
test/pytest_sample.py::test_fun2_subtraction[5-3--2] PASSED                                                                                                                                                                                     [ 29%]
test/pytest_sample.py::test_fun2_subtraction[3-5-2] PASSED                                                                                                                                                                                      [ 33%]
test/pytest_sample.py::test_fun2_subtraction[-2--3--1] PASSED                                                                                                                                                                                   [ 37%]
test/pytest_sample.py::test_fun2_subtraction[0-0-0] PASSED                                                                                                                                                                                      [ 41%]
test/pytest_sample.py::test_fun2_subtraction[10--5--15] PASSED                                                                                                                                                                                  [ 45%]
test/pytest_sample.py::test_fun2_invalid_types PASSED                                                                                                                                                                                           [ 50%]
test/pytest_sample.py::test_fun3_multiplication[2-3-6] PASSED                                                                                                                                                                                   [ 54%]
test/pytest_sample.py::test_fun3_multiplication[-2-3--6] PASSED                                                                                                                                                                                 [ 58%]
test/pytest_sample.py::test_fun3_multiplication[-3--5-15] PASSED                                                                                                                                                                                [ 62%]
test/pytest_sample.py::test_fun3_multiplication[0-10-0] PASSED                                                                                                                                                                                  [ 66%]
test/pytest_sample.py::test_fun3_invalid_types PASSED                                                                                                                                                                                           [ 70%]
test/pytest_sample.py::test_fun4_combined_math[2-3] PASSED                                                                                                                                                                                      [ 75%]
test/pytest_sample.py::test_fun4_combined_math[-1-2] PASSED                                                                                                                                                                                     [ 79%]
test/pytest_sample.py::test_fun4_combined_math[-4--5] PASSED                                                                                                                                                                                    [ 83%]
test/pytest_sample.py::test_fun4_combined_math[0-0] PASSED                                                                                                                                                                                      [ 87%]
test/pytest_sample.py::test_fun4_invalid_types PASSED                                                                                                                                                                                           [ 91%]
test/pytest_sample.py::test_fun4_is_consistent_with_individual_functions PASSED                                                                                                                                                                 [ 95%]
test/pytest_sample.py::test_zero_behavior PASSED                                                                                                                                                                                                [100%]

================================================================================================================= 24 passed in 0.04s ==================================================================================================================
(MLOps) rjaditya@Adityas-MacBook-Air LAB3 % 
```

**Unittest Results**
```
(MLOps) rjaditya@Adityas-MacBook-Air LAB3 % python test/unittest_sample.py 
..........
----------------------------------------------------------------------
Ran 10 tests in 0.000s

OK
(MLOps) rjaditya@Adityas-MacBook-Air LAB3 % python test/unittest_sample.py -v
test_fun1_addition (__main__.TestCalculator) ... ok
test_fun1_invalid_types (__main__.TestCalculator) ... ok
test_fun2_invalid_types (__main__.TestCalculator) ... ok
test_fun2_subtraction (__main__.TestCalculator) ... ok
test_fun3_invalid_types (__main__.TestCalculator) ... ok
test_fun3_multiplication (__main__.TestCalculator) ... ok
test_fun4_combined_math (__main__.TestCalculator) ... ok
test_fun4_consistency (__main__.TestCalculator) ... ok
test_fun4_invalid_types (__main__.TestCalculator) ... ok
test_zero_behavior (__main__.TestCalculator) ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.000s

OK
(MLOps) rjaditya@Adityas-MacBook-Air LAB3 % 
```

### STEP5: GitHub Actions
- Created yml files to run pytest and unittest scripts
- yml files present in `MLOPS_rjaditya/.github/workflows`
- tests will run only when there is a change to LAB3