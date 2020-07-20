import json
from typing import Optional, List, Union

from dataclasses import dataclass
import dataclasses
from ai_harness.buffer import *


@dataclass
class TaskInputExample:
    guid: str
    text_a: Optional[Union[str, List[str]]]
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class TaskInputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    guid: int = None

    def fields_tuple(self):
        return [self.input_ids, self.attention_mask, self.token_type_ids, self.label, self.guid]

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class FeaturesSerializer(ObjectListSerializer):
    def __init__(self, dest_file):
        self._buf = open(dest_file, 'wb+')
        super().__init__([L_Int, L_Int, L_Int, T_Int, T_Int], self._buf)

    def write_features(self, features: TaskInputFeatures):
        super().write(*features.fields_tuple())

    def write_features_list(self, features: List[TaskInputFeatures]):
        for f in features: self.write_features(f)

    def read_features_iter(self):
        for fields in super().read_iter():
            yield TaskInputFeatures(input_ids=fields[0],
                                    attention_mask=fields[1],
                                    token_type_ids=fields[2],
                                    label=fields[3],
                                    guid=fields[4]
                                    )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._buf: self._buf.close()
