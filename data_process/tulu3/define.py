from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal

from bson.objectid import ObjectId


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> 'Message':
        return cls(**obj)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TuluInstance:
    # 由于 tulu.id 有重复，所以额外使用 uuid 作为唯一标志
    id: str
    messages: List[Message]
    source: str
    uuid: ObjectId = field(default_factory=ObjectId)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> 'TuluInstance':
        if "uuid" in obj:
            obj["uuid"] = ObjectId(obj["uuid"])
        obj["messages"] = [Message.from_dict(obj=m) for m in obj.pop("messages")]
        return cls(**obj)

    def to_dict(self) -> Dict[str, Any]:
        obj = asdict(self)
        obj["uuid"] = str(self.uuid)
        return obj


@dataclass
class MessageWithChunks:
    role: Literal["system", "user", "assistant"]
    content: str
    chunks: List[str]

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "MessageWithChunks":
        return cls(**obj)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TuluInstanceWithChunks:
    id: str
    messages: List[MessageWithChunks]
    source: str
    uuid: ObjectId

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "TuluInstanceWithChunks":
        messages = [MessageWithChunks.from_dict(obj=m) for m in obj.pop("messages")]
        uuid = ObjectId(obj.pop("uuid"))
        return cls(messages=messages, uuid=uuid, **obj)

    def to_dict(self) -> Dict[str, Any]:
        obj = asdict(self)
        obj["uuid"] = str(self.uuid)
        return obj


@dataclass
class SFTInputs:
    labels: List[int]
    input_ids: List[int]

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> 'SFTInputs':
        return cls(**obj)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SFTInstance:
    uuid: ObjectId
    tulu_uuid: ObjectId
    prompt: str
    response: str
    inputs: SFTInputs

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> 'SFTInstance':
        if "uuid" in obj:
            obj["uuid"] = ObjectId(obj["uuid"])
        if "tulu_uuid" in obj:
            obj["tulu_uuid"] = ObjectId(obj["tulu_uuid"])
        inputs = SFTInputs.from_dict(obj.pop("inputs"))
        return cls(inputs=inputs, **obj)

    def to_dict(self) -> Dict[str, Any]:
        obj = asdict(self)
        obj["uuid"] = str(self.uuid)
        obj["tulu_uuid"] = str(self.tulu_uuid)
        return obj


@dataclass
class SFTInstanceWithChunks:
    uuid: ObjectId
    tulu_uuid: ObjectId
    prompt: str
    response: str
    chunks: List[str]
    inputs: SFTInputs
    block_inputs: SFTInputs
    block_tokens: List[int]
    response_tokens: int

    train_block: bool

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> 'SFTInstanceWithChunks':
        uuid = ObjectId(obj.pop("uuid"))
        tulu_uuid = ObjectId(obj.pop("tulu_uuid"))
        inputs = SFTInputs.from_dict(obj=obj.pop("inputs"))
        return cls(uuid=uuid, tulu_uuid=tulu_uuid, inputs=inputs, **obj)

    def to_dict(self, skip_inputs: bool = False) -> Dict[str, Any]:
        if skip_inputs:
            self.inputs.input_ids = []
            self.inputs.labels = []
        obj = asdict(self)
        obj["uuid"] = str(self.uuid)
        obj["tulu_uuid"] = str(self.tulu_uuid)
        return obj
