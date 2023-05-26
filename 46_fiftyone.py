# @Author  : YashowHoo
# @File    : 46_fiftyone.py
# @Description : fiftyone quick start

import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
session.wait()

