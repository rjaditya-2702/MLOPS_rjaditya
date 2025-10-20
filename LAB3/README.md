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


