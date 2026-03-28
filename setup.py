from setuptools import setup, find_packages

setup(
    name="asl_cslr",
    version="0.1.0",
    description="ASL Continuous Sign Language Recognition using skeleton-based models",
    author="Vishruth",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "mediapipe>=0.10.33,<0.11",
        "opencv-python>=4.8",
        "Flask>=3.0",
        "Flask-SocketIO>=5.3",
        "simple-websocket>=1.0",
        "numpy>=1.24",
        "h5py>=3.9",
        "pandas>=2.0",
        "tqdm>=4.65",
        "tensorboard>=2.14",
        "pyyaml>=6.0",
        "editdistance>=0.6",
    ],
    entry_points={
        "console_scripts": [
            "asl-preprocess=scripts.preprocess:main",
            "asl-train=scripts.train:main",
            "asl-evaluate=scripts.evaluate:main",
            "asl-online=scripts.run_online:main",
        ],
    },
)
