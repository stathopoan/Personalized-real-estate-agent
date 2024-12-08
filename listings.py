import csv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from models import RealEstate


def create_listings(chat_model_name):
    temperature = 0.3

    chat_llm = ChatOpenAI(model_name=chat_model_name, temperature=temperature, max_tokens=4096)

    parser = JsonOutputParser(pydantic_object=RealEstate)

    user_query = '''You are populating a database for a real estate agent. We need listings, 1 listing for every real 
    estate. Create listings of real estates with the following attributes: neighborhood, price, bedrooms, bathrooms, 
    house_size, description, neighborhood_description. Use your imagination to synthesize 
    description (regarding the house) and neighborhood_description (regarding the neighborhood and the area around the house). 
    The descriptions for each listing must be different.

    Example:  

    {
    "neighborhood": "Green Oaks", 
    "price": 800000,
    "bedrooms": 3, 
    "bathrooms": 2, 
    "house_size": 2000,
    "description": "Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.",  
    "neighborhood_description": "Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze"
    }
    
    We need a list of jsons in the following format:
    [{},{},...{}]. We expect a list of 20 listings
    '''

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    print("=== Chat Response ===")
    answer = chain.invoke({"query": user_query})
    print (answer)
    headers = answer[0].keys()
    with open('listings.csv', 'w', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(answer)