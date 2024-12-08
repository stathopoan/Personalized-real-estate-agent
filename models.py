import vector
from pydantic import BaseModel, Field
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

# func = get_registry().get("openai").create(name="text-embedding-3-small")
class RealEstate(BaseModel):
    neighborhood: str = Field(description="The neighborhood of the real estate")
    price: int = Field(description="The buying price in dollars $")
    bedrooms: int = Field(description="The number of bedrooms in the real estate")
    bathrooms: int = Field(description="The number of bathrooms in the real estate")
    house_size: int = Field(description="How big is the real estate in square feet (sqft)")
    description: str = Field(description="A brief description of the real estate including useful information for the "
                                         "buyer")
    neighborhood_description: str = Field(description="A brief description of the area the real estate is located")


class RealEstateDB(LanceModel):
    neighborhood: str
    price: int
    bedrooms: int
    bathrooms: int
    house_size: int
    description: str
    neighborhood_description: str
    # vector: Vector(func.ndims()) = func.VectorField()
    vector: Vector(1536)
    text: str