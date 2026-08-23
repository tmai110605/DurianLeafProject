"""
models package
==============
Export all model architectures: multi-task and single-task.
"""

from models.MAFC import MAFC, ChannelGate, SpatialGate, BasicConv

from models.mobilenetv3_custom import (
    MobileNetV2MultiTask,
    MobileNetV3SmallMultiTask,
    MobileNetV3LargeMultiTask,
)

from models.mobilenetv3_single import (
    MobileNetV2SingleTask,
    MobileNetV3SmallSingleTask,
    MobileNetV3LargeSingleTask,
)

from models.resnet_custom import (
    ResNet18MultiTask,
    ResNet50MultiTask,
)

from models.resnet_single import (
    ResNet18SingleTask,
    ResNet50SingleTask,
)

from models.densenet_custom import DenseNet121MultiTask

from models.densenet_single import DenseNet121SingleTask

from models.ticknet_custom import TickNetLargeMultiTask

from models.ticknet_single import TickNetLargeSingleTask

__all__ = [
    # Attention module
    "MAFC", "ChannelGate", "SpatialGate", "BasicConv",
    # Multi-task
    "MobileNetV2MultiTask", "MobileNetV3SmallMultiTask", "MobileNetV3LargeMultiTask",
    "ResNet18MultiTask", "ResNet50MultiTask",
    "DenseNet121MultiTask",
    "TickNetLargeMultiTask",
    # Single-task
    "MobileNetV2SingleTask", "MobileNetV3SmallSingleTask", "MobileNetV3LargeSingleTask",
    "ResNet18SingleTask", "ResNet50SingleTask",
    "DenseNet121SingleTask",
    "TickNetLargeSingleTask",
]
