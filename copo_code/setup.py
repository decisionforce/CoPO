# Please don't change the order of following packages!
import sys
from distutils.core import setup

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

setup(
    name="copo",
    install_requires=[
        "yapf==0.30.0",
        "ray==2.2.0",
        "ray[rllib]==2.2.0",
        "tensorflow==2.3.1",
        "torch",
        "tensorflow-probability==0.11.1",
        "tensorboardX",
        "gym==0.19.0"
    ],
    license="Apache 2.0",
)
