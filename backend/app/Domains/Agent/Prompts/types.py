from typing import Dict, List, TypeAlias, TypedDict


class NodeContent(TypedDict):
    role: str
    content: str


NodeMessage: TypeAlias = Dict[str, List[NodeContent]]
