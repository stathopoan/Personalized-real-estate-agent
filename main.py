import json
import os
import lancedb
import pandas as pd
import pyarrow as pa

from listings import create_listings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader

from models import RealEstateDB

os.environ["OPENAI_API_KEY"] = "voc-850085367126677217758866f45683d08462.60723152"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"
chat_model_name = "gpt-3.5-turbo"
table_name = "listings"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

chat_llm = ChatOpenAI(model_name=chat_model_name, temperature=0.0, max_tokens=1000)


# client = OpenAI()

# def get_embedding(text, model="text-embedding-3-small"):
#    text = text.replace("\n", " ")
#    return client.embeddings.create(input = [text], model=model).data[0].embedding

# def get_embedding(text):
#     text = text.replace("\n", " ")
#     return embeddings_model.embed_query(text)

def prepare_text_for_embedding(row):
    '''
    Concatenate all attributes into one big string
    :param row: The row of the listings dataframe
    :return: the concatenated string
    '''
    return "neighborhood: " + row['neighborhood'] + ",price: " + str(row['price']) + ",bedrooms: " + str(
        row['bedrooms']) + ",bathrooms: " + str(row['bathrooms']) + ",house_size: " + str(
        row['house_size']) + "sqft" + ",description: " + row['description'] + ",neighborhood_description: " + row[
        'neighborhood_description']


def generate_embeddings(text):
    '''
    Create embeddings for a specific text
    :param text: The text to be converted
    :return: Teh actual embeddings
    '''
    text = text.replace("\n", " ")
    return embeddings_model.embed_query(text)


def create_and_populate_db(table_name):
    # loader = CSVLoader("listings.csv")
    # data_docs = loader.load()
    # docs = [d.page_content for d in data_docs]
    # embeddings = embeddings_model.embed_documents(docs)

    # df_text = pd.DataFrame(docs, columns=["text"])
    df_listings_attr = pd.read_csv("listings.csv")
    df_listings_attr['text'] = df_listings_attr.apply(prepare_text_for_embedding,
                                                      axis=1)  # Add text column including all attributes in text
    df_listings_attr['vector'] = df_listings_attr['text'].apply(generate_embeddings)  # The embeddings of the text field

    db = lancedb.connect("~/.lancedb")
    db.drop_table(table_name, ignore_missing=True)

    data = pa.Table.from_pandas(df_listings_attr)
    table = db.create_table(table_name, schema=RealEstateDB, data=data)


def get_user_preferences():
    questions = [
        "How big do you want your house to be?"
        "What are 3 most important things for you in choosing this property?",
        "Which amenities would you like?",
        "Which transportation options are important to you?",
        "How urban do you want your neighborhood to be?",
    ]
    answers = [
        "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
        "A quiet neighborhood, good local schools, and convenient shopping options.",
        "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
        "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
        "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
    ]

    customer_preferences = ""
    for answer in answers:
        customer_preferences = customer_preferences + " " + answer

    return customer_preferences

def query_db(table_name, query_text, apply_user_preferences=True):
    db = lancedb.connect("~/.lancedb")
    table = db.open_table(table_name)

    query = query_text
    if apply_user_preferences:
        user_preferences = get_user_preferences()
        query = query_text + "as long as it satisfies these criteria: " + user_preferences

    query_vector = generate_embeddings(query)
    found_estates = table.search(query_vector).limit(3).to_pydantic(RealEstateDB)
    for e in found_estates:
        print(json.dumps(e.dict(exclude={'vector','text'}), indent=1))
        # print(e.model_dump(exclude={'vector','text'}))


# create_listings(chat_model_name)
# create_and_populate_db(table_name)
query_db(table_name, "I want a real estate that costs less than 90000$", apply_user_preferences=False)
