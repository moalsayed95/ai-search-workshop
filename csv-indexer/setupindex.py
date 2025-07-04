import os
import glob
import pandas as pd
import json
import azure.search.documents

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient  
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from azure.search.documents.indexes.models import (
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    AzureOpenAIEmbeddingSkill,
    PIIDetectionSkill,
    SearchIndexerSkillset,
    SearchIndexer,
    FieldMapping,
    FieldMappingFunction,
    IndexingParameters,
    IndexingParametersConfiguration,
    BlobIndexerParsingMode,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataContainer,
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    VectorSearchAlgorithmKind,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    VectorSearchVectorizer,
    SearchFieldDataType
)

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from pydantic import BaseModel, Field
from typing import Optional


def setup_azure_openai_client(config):
    """Set up Azure OpenAI client with managed identity authentication."""
    try:        
        # Set up token provider for Azure OpenAI authentication
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), 
            "https://cognitiveservices.azure.com/.default"
        )
        
        # Create Azure OpenAI client with managed identity
        client = AzureOpenAI(
            api_version=config['azure_openai_api_version'],
            azure_endpoint=config['azure_openai_endpoint'],
            azure_ad_token_provider=token_provider  # Using managed identity, no API key
        )
         
        print(f"✅ Azure OpenAI client initialized with managed identity")
        
        return client
    except Exception as e:
        print(f"❌ Failed to initialize Azure OpenAI client: {e}")
        return None


def load_configuration():
    """Load configuration from environment variables and initialize credentials."""
    # Load environment variables from .env file
    load_dotenv(override=True)

    # Initialize Azure credentials for managed identity authentication
    credential = DefaultAzureCredential()

    # Azure AI Search configuration - NO API KEY
    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
    index_name = os.getenv("AZURE_SEARCH_INDEX", "csvvec-managed-identity")

    # Azure OpenAI configuration - NO API KEY
    azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    # Clean endpoint - remove trailing slash if present
    if azure_openai_endpoint.endswith('/'):
        azure_openai_endpoint = azure_openai_endpoint[:-1]
    
    azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    azure_openai_model_name = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-large")
    azure_openai_model_dimensions = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", 1024))
    azure_openai_chat_deployment = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "gpt-4o")
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    # Azure AI Services configuration - NO API KEY
    azure_ai_services_subdomain_url = os.environ["AZURE_AI_SERVICES_SUBDOMAIN_URL"]

    # Blob Storage configuration
    blob_connection_string = os.environ["BLOB_CONNECTION_STRING"]
    blob_container_name = os.getenv("BLOB_CONTAINER_NAME", "csv-vec")

    config = {
        'credential': credential,
        'endpoint': endpoint,
        'index_name': index_name,
        'azure_openai_endpoint': azure_openai_endpoint,
        'azure_openai_embedding_deployment': azure_openai_embedding_deployment,
        'azure_openai_model_name': azure_openai_model_name,
        'azure_openai_model_dimensions': azure_openai_model_dimensions,
        'azure_openai_chat_deployment': azure_openai_chat_deployment,
        'azure_openai_api_version': azure_openai_api_version,
        'azure_ai_services_subdomain_url': azure_ai_services_subdomain_url,
        'blob_connection_string': blob_connection_string,
        'blob_container_name': blob_container_name
    }

    return config


def print_configuration(config):
    """Print the loaded configuration for verification."""
    print("✅ Configuration loaded successfully:")
    print(f"🔗 Azure AI Search endpoint: {config['endpoint']}")
    print(f"🔗 Azure OpenAI endpoint: {config['azure_openai_endpoint']}")
    print(f"🔗 Azure AI Services subdomain: {config['azure_ai_services_subdomain_url']}")
    print(f"📦 Blob container: {config['blob_container_name']}")
    print(f"🔍 Search index: {config['index_name']}")
    print("🔐 Using managed identity for all authentication - NO API KEYS!")
    print(f"Azure AI Search SDK Version: {azure.search.documents.__version__}")


def verify_authentication(config):
    """Verify we can authenticate to Azure AI Search."""
    try:
        index_client = SearchIndexClient(endpoint=config['endpoint'], credential=config['credential'])
        # Try to list indexes to verify authentication
        indexes = list(index_client.list_indexes())
        print(f"🔍 Found {len(indexes)} existing indexes")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False


def upload_sample_documents(config):
    """Upload CSV files to blob storage using managed identity."""
    import re
    
    # Extract the storage account URL from the connection string
    connection_string = config['blob_connection_string']
    account_name_match = re.search(r'AccountName=([^;]+)', connection_string)
    
    if account_name_match:
        account_name = account_name_match.group(1)
        account_url = f"https://{account_name}.blob.core.windows.net"
    else:
        # Fallback - try to get from environment or construct
        account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
        if not account_url:
            raise ValueError("Could not determine storage account URL. Please set AZURE_STORAGE_ACCOUNT_URL environment variable.")
    
    print(f"🔐 Using managed identity for blob storage access: {account_url}")
    print(f"   💡 Storage account has key-based auth disabled (secure setup)")
    
    # Use managed identity for blob storage
    blob_service_client = BlobServiceClient(
        account_url=account_url,
        credential=config['credential']
    )
    
    container_client = blob_service_client.get_container_client(config['blob_container_name'])
    
    try:
        if not container_client.exists():
            container_client.create_container()
            print(f"📦 Created container: {config['blob_container_name']}")
    except Exception as e:
        print(f"⚠️ Container check/creation failed: {e}")
        print("💡 Make sure your user identity has 'Storage Blob Data Contributor' role on the storage account")

    documents_directory = "csv_data"
    csv_files = glob.glob(os.path.join(documents_directory, '*.csv'))
    
    if not csv_files:
        print(f"⚠️ No CSV files found in {documents_directory}")
        return False
    
    uploaded_count = 0
    for file in csv_files:
        with open(file, "rb") as data:
            name = os.path.basename(file)
            try:
                if not container_client.get_blob_client(name).exists():
                    container_client.upload_blob(name=name, data=data)
                    print(f"📄 Uploaded: {name}")
                    uploaded_count += 1
                else:
                    print(f"📄 Already exists: {name}")
            except Exception as e:
                print(f"❌ Failed to upload {name}: {e}")

    print(f"✅ Sample data setup completed in {config['blob_container_name']} ({uploaded_count} files uploaded)")
    return True


def create_embedding_skills(config):
    """Create Azure OpenAI embedding skills."""
    # Azure OpenAI Embedding Skill for Title - NO API KEY
    title_embedding_skill = AzureOpenAIEmbeddingSkill(  
        description="Title embeddings via Azure OpenAI with managed identity",  
        context="/document",  
        resource_url=config['azure_openai_endpoint'],
        deployment_name=config['azure_openai_embedding_deployment'],
        model_name=config['azure_openai_model_name'],
        dimensions=config['azure_openai_model_dimensions'],
        inputs=[  
            InputFieldMappingEntry(name="text", source="/document/Title"),  
        ],  
        outputs=[  
            OutputFieldMappingEntry(name="embedding", target_name="TitleVector")  
        ],  
    )

    # Azure OpenAI Embedding Skill for Description - NO API KEY
    description_embedding_skill = AzureOpenAIEmbeddingSkill(  
        description="Description embeddings via Azure OpenAI with managed identity",  
        context="/document",  
        resource_url=config['azure_openai_endpoint'],
        deployment_name=config['azure_openai_embedding_deployment'],
        model_name=config['azure_openai_model_name'],
        dimensions=config['azure_openai_model_dimensions'],
        inputs=[  
            InputFieldMappingEntry(name="text", source="/document/Description"),  
        ],  
        outputs=[  
            OutputFieldMappingEntry(name="embedding", target_name="DescriptionVector")  
        ],  
    )

    return title_embedding_skill, description_embedding_skill


def create_pii_skill(config):
    """Create PII detection skill."""
    pii_skill = PIIDetectionSkill(
        name="mask-description-pii",
        description="PII detection using managed identity",
        context="/document",
        default_language_code="en",
        minimum_precision=0.5,
        masking_mode="replace",
        maskingCharacter="*",  # This parameter may show warning but skillset still works
        inputs=[
            InputFieldMappingEntry(name="text", source="/document/Description")
        ],
        outputs=[
            OutputFieldMappingEntry(name="maskedText", target_name="DescriptionRedacted")
        ]
    )
    return pii_skill


def create_skillset(config):
    """Create and deploy the skillset with all skills."""
    skillset_name = f"{config['index_name']}-skillset"
    
    # Create skills
    title_embedding_skill, description_embedding_skill = create_embedding_skills(config)
    pii_skill = create_pii_skill(config)
    
    # Combine all skills
    skills = [title_embedding_skill, description_embedding_skill, pii_skill]

    # Create skillset without AIServicesAccountIdentity (which was causing InvalidApiType error)
    skillset = SearchIndexerSkillset(  
        name=skillset_name,  
        description="Embeddings + PII detection using managed identity",  
        skills=skills
    )

    # Deploy skillset
    try:
        client = SearchIndexerClient(config['endpoint'], config['credential'])
        client.create_or_update_skillset(skillset)  
        print(f"✅ Skillset '{skillset.name}' created/updated successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create skillset: {e}")
        if "403" in str(e) or "Forbidden" in str(e):
            print("💡 Ensure your search service has required RBAC roles:")
            print("   - 'Cognitive Services OpenAI User' on Azure OpenAI resource")
            print("   - 'Cognitive Services User' on Azure AI Services resource")
        return False


def create_search_index(config):
    """Create the search index with vector fields."""
    index_name = config['index_name']
    
    # Create vector search configuration
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
                vectorizer_name="myOpenAI"
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="myOpenAI",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=config['azure_openai_endpoint'],
                    deployment_name=config['azure_openai_embedding_deployment'],
                    model_name=config['azure_openai_model_name']
                )
            )
        ]
    )

    # Define the fields for the index
    fields = [
        SimpleField(name="ID", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
        SearchableField(name="Name", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
        SimpleField(name="Age", type=SearchFieldDataType.Int32, sortable=True, filterable=True, facetable=True),
        SearchableField(name="Title", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
        SearchableField(name="Description", type=SearchFieldDataType.String, sortable=True),
        SearchableField(name="DescriptionRedacted", type=SearchFieldDataType.String, sortable=True),
        SearchField(
            name="TitleVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=config['azure_openai_model_dimensions'],
            vector_search_profile_name="myHnswProfile"
        ),
        SearchField(
            name="DescriptionVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=config['azure_openai_model_dimensions'],
            vector_search_profile_name="myHnswProfile"
        )
    ]

    # Create the search index
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    
    try:
        index_client = SearchIndexClient(endpoint=config['endpoint'], credential=config['credential'])
        result = index_client.create_or_update_index(index)
        print(f"✅ Search index '{result.name}' created/updated successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create search index: {e}")
        return False


def create_data_source(config):
    """Create the data source for the indexer using original connection string."""
    data_source_name = f"{config['index_name']}-datasource"
    container = SearchIndexerDataContainer(name=config['blob_container_name'])
    
    # Use original connection string (with storage keys)
    data_source = SearchIndexerDataSourceConnection(
        name=data_source_name,
        type="azureblob",
        connection_string=config['blob_connection_string'],  # Use original
        container=container
    )
    
    try:
        indexer_client = SearchIndexerClient(config['endpoint'], config['credential'])
        result = indexer_client.create_or_update_data_source_connection(data_source)
        print(f"✅ Data source '{result.name}' created successfully")
        return data_source_name
    except Exception as e:
        print(f"❌ Failed to create data source: {e}")
        return None

# def create_data_source(config):
#     """Create the data source for the indexer - trying managed identity first, then falling back to keys."""
#     data_source_name = f"{config['index_name']}-datasource"
    
#     # Extract storage account name from connection string
#     import re
#     connection_string = config['blob_connection_string']
#     account_name_match = re.search(r'AccountName=([^;]+)', connection_string)
    
#     if account_name_match:
#         account_name = account_name_match.group(1)
#         # Try managed identity connection string format
#         managed_identity_connection_string = f"BlobEndpoint=https://{account_name}.blob.core.windows.net/"
        
#         print(f"   🔧 Account name: {account_name}")
#         print(f"   📝 Trying managed identity format first, then falling back to key-based")
#     else:
#         print("⚠️ Could not extract storage account name. Using original connection string.")
#         managed_identity_connection_string = None
    
#     container = SearchIndexerDataContainer(name=config['blob_container_name'])
    
#     # Use managed identity for data source (storage has key-based auth disabled)
#     if managed_identity_connection_string:
#         print(f"🔐 Using managed identity connection for data source")
#         print(f"   Connection: {managed_identity_connection_string}")
#         print(f"   💡 Storage account requires managed identity (key-based auth disabled)")
        
#         data_source = SearchIndexerDataSourceConnection(
#             name=data_source_name,
#             type="azureblob",
#             connection_string=managed_identity_connection_string,
#             container=container
#         )
        
#         try:
#             indexer_client = SearchIndexerClient(config['endpoint'], config['credential'])
#             result = indexer_client.create_or_update_data_source_connection(data_source)
#             print(f"✅ Data source '{result.name}' created successfully with managed identity")
#             return data_source_name
#         except Exception as e:
#             print(f"❌ Managed identity failed: {str(e)[:200]}...")
#             print(f"   💡 This likely means the search service's managed identity needs:")
#             print(f"   1. System-assigned managed identity enabled on search service")
#             print(f"   2. 'Storage Blob Data Reader' role on storage account for search service identity")
#             return None
#     else:
#         print(f"❌ Could not create managed identity connection string")
#         return None


def create_indexer(config):
    """Create and run the indexer."""
    index_name = config['index_name']
    skillset_name = f"{index_name}-skillset"
    indexer_name = f"{index_name}-indexer"
    data_source_name = f"{index_name}-datasource"
    
    # Create indexer parameters for CSV processing
    indexer_parameters = IndexingParameters(
        configuration=IndexingParametersConfiguration(
            parsing_mode=BlobIndexerParsingMode.DELIMITED_TEXT,
            query_timeout=None,
            first_line_contains_headers=True
        )
    )

    # Create the indexer
    indexer = SearchIndexer(  
        name=indexer_name,  
        description="Indexer to index documents and generate embeddings",  
        skillset_name=skillset_name,  
        target_index_name=index_name,  
        data_source_name=data_source_name,
        parameters=indexer_parameters,
        field_mappings=[
            FieldMapping(
                source_field_name="AzureSearch_DocumentKey", 
                target_field_name="ID", 
                mapping_function=FieldMappingFunction(name="base64Encode")
            )
        ],
        output_field_mappings=[
            FieldMapping(source_field_name="/document/TitleVector", target_field_name="TitleVector"),
            FieldMapping(source_field_name="/document/DescriptionVector", target_field_name="DescriptionVector"),
            FieldMapping(source_field_name="/document/DescriptionRedacted", target_field_name="DescriptionRedacted"),
        ]
    )

    try:
        indexer_client = SearchIndexerClient(config['endpoint'], config['credential'])
        indexer_result = indexer_client.create_or_update_indexer(indexer)
        print(f"✅ Indexer '{indexer_result.name}' created/updated successfully")
        
        # Run the indexer  
        indexer_client.run_indexer(indexer_name)  
        print(f"🚀 Indexer '{indexer_name}' is running. Processing documents...")
        return True
    except Exception as e:
        print(f"❌ Failed to create/run indexer: {e}")
        return False


def perform_hybrid_search(config, query="trainer", top_results=3):
    """Perform hybrid search combining text and vector search."""
    
    search_client = SearchClient(
        config['endpoint'], 
        config['index_name'], 
        credential=config['credential']
    )

    # Create vector query for hybrid search
    vector_query = VectorizableTextQuery(
        text=query, 
        k_nearest_neighbors=50, 
        fields="TitleVector,DescriptionVector"
    )
    
    try:
        results = search_client.search(  
            search_text=query,  
            vector_queries=[vector_query],
            select=["ID", "Name", "Title", "DescriptionRedacted"],
            top=top_results
        )
        
        print(f"\n🔍 Hybrid search results for '{query}':")
        print("=" * 50)
        
        result_count = 0
        for result in results:
            result_count += 1
            print(f"\n📋 Result {result_count}:")
            print(f"   Score: {result['@search.score']:.4f}")  
            print(f"   ID: {result['ID']}")  
            print(f"   Name: {result['Name']}")  
            print(f"   Title: {result['Title']}")
            print(f"   Description: {result['DescriptionRedacted']}")
            
        if result_count == 0:
            print("   No results found. The indexer might still be processing documents.")
            print("   Please wait a few minutes and try again.")
            
        return result_count > 0
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False


def main():
    """Main function to orchestrate the setup process."""
    print("🚀 Starting Azure AI Search Managed Identity Setup")
    print("=" * 60)
    
    print("\n📋 Step 1: Loading configuration...")
    try:
        config = load_configuration()
        # print_configuration(config)
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

    # print("\n🔐 Step 2: Verifying authentication...")
    # if not verify_authentication(config):
    #     return False


    # print("\n🤖 Step 3: Setting up Azure OpenAI client...")
    # openai_client = setup_azure_openai_client(config)
    # if not openai_client:
    #     return False

    # print("\n📦 Step 4: Uploading sample documents...")
    # upload_sample_documents(config)

    # print("\n🧠 Step 5: Creating skillset...")
    # if not create_skillset(config):
    #     return False

    # print("\n🔍 Step 6: Creating search index...")
    # if not create_search_index(config):
    #     return False

    # print("\n📊 Step 7: Creating data source...")
    # data_source_name = create_data_source(config)
    # if not data_source_name:
    #     return False

    # print("\n⚙️ Step 8: Creating and running indexer...")
    # if not create_indexer(config):
    #     return False

    print("\n🔍 Step 9: Testing hybrid search...")
    print("⏳ Waiting a moment for indexer to process documents...")
    import time
    time.sleep(10) 
    
    perform_hybrid_search(config,query="trainer", top_results=3)

    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)