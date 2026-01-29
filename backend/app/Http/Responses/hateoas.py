from typing import List, Optional

from pydantic import BaseModel, Field


class Link(BaseModel):
    href: str
    method: str
    rel: str


class HateoasModel(BaseModel):
    # Field to be defined by subclasses to control order
    # links: List[Link] = Field(default_factory=list, alias="_links")

    def add_link(self, rel: str, href: str, method: str = "GET"):
        # Assumes subclass has 'links' field
        if not hasattr(self, "links"):
            # Fallback or error?
            # For Pydantic models, we can set attributes if config allows,
            # but better to expect the field exists.
            pass
        self.links.append(Link(rel=rel, href=href, method=method))

    def add_link(self, rel: str, href: str, method: str = "GET"):
        self.links.append(Link(rel=rel, href=href, method=method))
