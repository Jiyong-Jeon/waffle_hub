__version__ = "0.2.16"

import enum

from waffle_dough.type.data_type import DataType
from waffle_hub.type.backend_type import BackendType
from waffle_hub.utils.utils import CaseInsensitiveDict


class CustomEnumMeta(enum.EnumMeta):
    def __contains__(cls, item):
        if isinstance(item, str):
            return item.upper() in cls._member_names_
        return super().__contains__(item)

    def __upper__(self):
        return self.name.upper()


class BaseEnum(enum.Enum, metaclass=CustomEnumMeta):
    """Base class for Enum

    Example:
        >>> class Color(BaseEnum):
        >>>     RED = 1
        >>>     GREEN = 2
        >>>     BLUE = 3
        >>> Color.RED == "red"
        True
        >>> Color.RED == "RED"
        True
        >>> "red" in DataType
        True
        >>> "RED" in DataType
        True
    """

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name.upper() == other.upper()
        return super().__eq__(other)

    def __ne__(self, other):
        if isinstance(other, str):
            return self.name.upper() != other.upper()
        return super().__ne__(other)

    def __hash__(self):
        return hash(self.name.upper())

    def __str__(self):
        return self.name.upper()

    def __repr__(self):
        return self.name.upper()


class SplitMethod(BaseEnum):
    RANDOM = enum.auto()
    STRATIFIED = enum.auto()


EXPORT_MAP = CaseInsensitiveDict(
    {
        DataType.YOLO: "ULTRALYTICS",
        DataType.ULTRALYTICS: "ULTRALYTICS",
        DataType.COCO: "COCO",
        DataType.AUTOCARE_DLT: "AUTOCARE_DLT",
        DataType.TRANSFORMERS: "TRANSFORMERS",
    }
)

BACKEND_MAP = CaseInsensitiveDict(
    {
        BackendType.ULTRALYTICS: {
            "import_path": "waffle_hub.hub.adapter.ultralytics",
            "class_name": "UltralyticsHub",
            "adapter_import_path": "waffle_hub.hub.train.adapter.ultralytics.ultralytics",
            "adapter_class_name": "UltralyticsManager",
        },
        BackendType.AUTOCARE_DLT: {
            "import_path": "waffle_hub.hub.adapter.autocare_dlt",
            "class_name": "AutocareDLTHub",
            "adapter_import_path": "waffle_hub.hub.train.adapter.autocare_dlt.autocare_dlt",
            "adapter_class_name": "AutocareDltManager",
        },
        BackendType.TRANSFORMERS: {
            "import_path": "waffle_hub.hub.adapter.transformers",
            "class_name": "TransformersHub",
            "adapter_import_path": "waffle_hub.hub.train.adapter.transformers.transformers",
            "adapter_class_name": "TransformersManager",
        },
    }
)
