from setuptools import setup, find_packages

# Read requirements.txt, ignore comments
try:
    REQUIRES = list()
    f = open("requirements.txt", "rb")
    for line in f.read().decode("utf-8").split("\n"):
        line = line.strip()
        if "#" in line:
            line = line[:line.find("#")].strip()
        if line:
            REQUIRES.append(line)
except:
    print("'requirements.txt' not found!")
    REQUIRES = list()

setup(
    name = "rlzoo",
    version = "1.0.4",
    include_package_data=True,
    author='Zihan Ding, Tianyang Yu, Yanhua Huang, Hongming Zhang, Hao Dong',
    author_email='zhding@mail.ustc.edu.cn',
    url = "https://github.com/tensorlayer/RLzoo" ,
    license = "apache" ,
    packages = find_packages(),
    install_requires=REQUIRES,
    description = "A collection of reinforcement learning algorithms with hierarchical code structure and convenient APIs.",
    keywords = "Reinforcment Learning",
    platform=['any'],
    python_requires='>=3.5',
)
