# ============================================================
# Dependencies
# ============================================================

# read environment variables (OPENAI_API_KEY)
# build file paths safely
import os

# log server startup, errors, and status
# debug issues when server runs over stdio
import logging

# load secrets from .env into environment variables
# keep API keys out of source code
import dotenv

# mcp core types (protocol objects)
# tool -> describe callable actions (query_document)
# resource -> describe readable assets (PDF documents)
# TextContent -> text response from tool/resource to client
# ImageContent -> image response from tool/resource to client (not used here)
# EmbeddedResource -> reference to resource embedded in content (not used here)
# prompt -> describe resuable prompt templates (deep_analysis)
# PromptMessage -> role based messages in a prompt (system, user, assistant)
# PromptArgument -> arguments for prompts (query, info_type)
# GetPromptResult -> response from get_prompt handler, returned prmpt structure
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    GetPromptResult,
    Prompt,
    PromptMessage,
    PromptArgument
)

# MCP server runtime

# Server -> main MCP server class instance
# NotificationOptions -> controls server side notification/capabilities
from mcp.server import Server, NotificationOptions

# passed during server.run to configure server metadata/capabilities
# Advertises server name, version, capabilities to clients
from mcp.server.models import InitializationOptions

# runs the server over stdio streams
# enables mcp client <-> server communication without networking https
import mcp.server.stdio

# vector database (core rag functionality)
# connects to your persisted vector db
# performs similarity search on embedded chunks
import chromadb

# coverts text -> embeddings using OpenAI models
# must match embedding model used during collection setup
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# file and metadata helpers
# scans filesystem (./testing/*.pdf) for available documents
# used to dynamically list available resources in the server
import glob

# reads package version dynamically
# used to set server_version during server initialization
from importlib import metadata

# pdf loading
# extract text from pdf pages
# used when client requests read_resource for a pdf document
from langchain_community.document_loaders import PyPDFLoader

# ============================================================
# Logging
# ============================================================

# sets minimum log level to INFO, means DEBUG logs are ignored
# ensures logs actually appear in terminal when server runs over stdio
logging.basicConfig(level=logging.INFO)

# creates a named logger for this server
# name helps identify logs when multiple MCP servers are running, filter logs later 
# used like logger.info("connected to chromadb"), logger.error("failed to load pdf"), etc.
# logging is important for debugging MCP servers running over stdio unlike ui
logger = logging.getLogger("document-search-mcp")

# ============================================================
# Environment Variables
# ============================================================

# Load environment variables from .env file
# reads a local .env file, loads variables into os.environ or os.getenv
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================
# Initialize server
# ============================================================

# creates an MCP server instance with the name "document-search"
# name is advertised to clients during initialization, used for logging, helpful when multiple servers run
# this server object is used to register tool/resource/prompt, handle tool calls, run the mcp protocol loop
server = Server("document-search")

# ============================================================
# Initialize ChromaDB client
# ============================================================

# will hold the chromadb client (connection to vector db on disk)
client = None
# will hold the OpenAI embedding function (text -> embeddings)
embedding_function = None
# will hold the chromadb collection (pdf_collection) that stores document embeddings
collection = None

# initialize everything required for vector search once at server startup ie connection to chromadb, embedding function, collection
try: 
    # connect to persisted chromadb instance on disk
    # must match path used during chroma_setup.ipynb
    # makes embeddings available after server starts
    client = chromadb.PersistentClient(path="./chroma_db")
    logger.info("Successfully connected to ChromaDB")
    
    # Initialize OpenAI embedding function
    # repeats the same embedding function used during ingestion
    # required so that query embeddings match stored document embeddings
    if OPENAI_API_KEY:
        embedding_function = OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
        logger.info("Successfully initialized OpenAI embedding function")
        
        # Get the collection
        # opens the existing collection
        # Binds the embedding function so queries are embedded correctly, search works as expected
        collection = client.get_collection(
            name="pdf_collection",
            embedding_function=embedding_function
        )
        logger.info(f"Successfully connected to collection with {collection.count()} documents")
    else:
        logger.warning("OpenAI API key is not set - embedding function not initialized")
except Exception as e:
    logger.error(f"Error initializing components: {e}")

# ============================================================
# Format search result helper function for query_document tool
# ============================================================

# utility function to format one search result, turns raw vector search output into llm friendly string
# inputs: document text, distance score, optional metadata dict
def format_search_result(document: str, distance: float, metadata: dict[str, object] = None) -> str:
    """Format a search result into a readable string."""
    # chroma returns distance where lower is better
    # convert to similarity score where higher is better
    # makes results easier to understand for users/llms
    result = f"Score: {1 - distance:.4f} (closer to 1 is better)\n"
    
    # page_num help grounding, citations, traceability
    if metadata:
        page_num = metadata.get('page', 'Unknown')
        result += f"Page: {page_num}\n"
    
    result += f"Content: {document}"
    return result

# ============================================================
# List available tools in the server
# ============================================================

# this function advertises what actions this mcp server can perform
# so any mcp client or llm can discover and use them
# this is capabilities advertisement
@server.list_tools() # register this function as the tool-list endpoint, mcp clients call this first to discover tools, without this clients won't know what tools are available
# async because mcp protocol is async
# returns a list of tool definitions not implementations
# This function tells the world: “Here are the tools I offer and how to call them.”
async def handle_list_tools() -> list[Tool]:
    """List available tools in the server."""
    return [
        # define the query_document tool
        Tool(
            # tool name the client will invoke
            # description helps the llm decide when to use this tool
            name="query_document",
            description="Search for information in the document based on semantic similarity",
            # json schema describing tool arguments
            # tells the llm what parameters are allowed and required
            # enables structured tool calls from llms instead of freeform text
            inputSchema={
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "The search query text"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)"
                    }
                },
                "required": ["query_text"]
            }
        ),
        # define the get_collection_info tool
        # lets client verify collection exists, how many documents are indexed
        Tool(
            name="get_collection_info",
            description="Get information about the ChromaDB collection",
            inputSchema={
                "type": "object",
                # tool takes no arguments
                "properties": {}
            }
        ),
    ]

# ============================================================
# Handle tool execution requests
# ============================================================

# tool execution router
# when the client says "call tool x with args y", mcp calls this function
@server.call_tool() # register this function as the handler for all tool calls coming from mcp clients
# name = which tool the client wants to call
# arguments = dict of arguments the client is passing to the tool
# returns list of mcp content objects (TextContent, ImageContent, EmbeddedResource) as the tool response, here we only use TextContent
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool execution requests."""
    
    # route to the correct tool based on name
    if name == "query_document":
        # extracts tool arguments with defaults
        query_text = arguments.get("query_text", "")
        num_results = arguments.get("num_results", 5)
        
       
        try:
            # Verify that ChromaDB and collection are properly initialized
            # validate db connection before querying, prevent crashing if setup failed
            if not collection:
                return [TextContent(
                    type="text",
                    text="Error: ChromaDB collection is not initialized. Please run chroma_setup.ipynb first."
                )]
                
            # Query the collection
            # chroma embeds the query text using the bound embedding function
            # performs similarity search to find closest document chunks
            results = collection.query(
                query_texts=[query_text],
                n_results=num_results
            )
            
            # Process results
            if not results or 'documents' not in results or not results['documents'][0]:
                return [TextContent(type="text", text="No results found for your query.")]
            
            formatted_results = []
            # format each result for better readability
            for i, (doc, distance, metadata) in enumerate(zip(
                results['documents'][0], 
                results['distances'][0],
                results['metadatas'][0] if 'metadatas' in results else [{}] * len(results['documents'][0])
            )):
                formatted_results.append(f"Result {i+1}:\n{format_search_result(doc, distance, metadata)}")
            
            # Return formatted results as a single TextContent object
            # mcp expects a list of content objects, here we return one with all results combined
            return [TextContent(
                type="text",
                text="\n\n---\n\n".join(formatted_results)
            )]
        
        except Exception as e:
            error_message = f"Error querying document: {str(e)}"
            logger.error(error_message)
            return [TextContent(type="text", text=error_message)]
    
    # handle get_collection_info tool
    elif name == "get_collection_info":
        try:
            if not collection:
                return [TextContent(
                    type="text",
                    text="Error: ChromaDB collection is not initialized. Please run chroma_setup.ipynb first."
                )]
                
            count = collection.count()
            # Return collection info
            return [TextContent(
                type="text",
                text=f"Collection name: pdf_collection\nNumber of documents: {count}"
            )]
        except Exception as e:
            error_message = f"Error getting collection info: {str(e)}"
            logger.error(error_message)
            return [TextContent(type="text", text=error_message)]
    else:
        raise ValueError(f"Unknown tool: {name}")

# ============================================================
# List available resources in the server
# ============================================================

# It advertises which documents are available on this MCP server as readable resources
# This is resource discovery.
@server.list_resources() # register this function as the resource-list endpoint, mcp clients call this to discover available resources/documents and how to reference them via uris
# async because mcp protocol is async
# returns a list of resource descriptors (metadata, not actual content)
# This function tells the client: “Here are the PDFs I have, and how to refer to them.”
async def handle_list_resources() -> list[Resource]:
    """List all available document resources"""
    resources = []

    # Find all available PDFs in testing directory
    try:
        # scan filesystem for pdfs
        pdf_files = glob.glob("./testing/*.pdf")
        # for each pdf found, create a Resource object
        for pdf_path in pdf_files:
            # extract filename without extension for uri and name
            filename = os.path.basename(pdf_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            resources.append(
                # create mcp Resource object
                # uri format: document://pdf/{document_name}, unique identifier the client uses to request the document
                # name -> human friendly name for the document
                # description -> brief description of the document
                # mimeType -> indicates this is a PDF document, helps clients handle it appropriately
                Resource(
                    uri=f"document://pdf/{name_without_ext}",
                    name=name_without_ext.replace('_', ' ').title(),
                    description=f"PDF Document: {name_without_ext}",
                    mimeType="application/pdf"
                )
            )
    except Exception as e:
        logger.error(f"Error scanning for PDF files: {e}")
    
    # client recieves this list of resources when it calls list_resources
    # later it can request the actual document content via read_resource using the uri
    return resources

# ============================================================
# Handle reading PDF resources
# ============================================================
# It returns the raw text of a PDF when a client asks to read a document resource.
# This complements vector search: search → find relevant chunks, read_resource → read the full source document
@server.read_resource() # register this function as the read_resource endpoint, mcp clients call this to read the actual content of a resource/document using its uri
async def handle_read_resource(uri: str):
    """Handle reading PDF resources"""
    
    # validate uri we received, ensure it matches expected format for pdf documents 
    if not str(uri).startswith("document://"):
        raise ValueError(f"Unsupported URI scheme: {uri}")
    
    # expecting format document://pdf/{document_name}
    path_parts = str(uri).split("/")
    if len(path_parts) < 4:
        raise ValueError(f"Invalid URI format: {uri}")
    
    # ensures only pdf documents are handled
    resource_type = path_parts[2]
    if resource_type != "pdf":
        raise ValueError(f"Unsupported resource type: {resource_type}")
        
    # extract document name from uri
    document_name = path_parts[3]
    
    try:            
        # Construct path - assuming documents are in ./testing/
        path = f"./testing/{document_name}"
        if not path.endswith('.pdf'):
            path += '.pdf'
            
        # Load the document
        loader = PyPDFLoader(path)
        pages = loader.load()
        
        # Combine all pages into one text with page markers
        full_text = ""
        for i, page in enumerate(pages):
            full_text += f"\n\n--- Page {i+1} ---\n\n"
            full_text += page.page_content
        logger.info(f"Loaded document {document_name} with {len(pages)} pages. Preview: {full_text[:200]}...")
        # Return the full text of the document
        return full_text
    except Exception as e:
        error_message = f"Error loading document: {str(e)}"
        logger.error(error_message)
        return error_message

# ============================================================
# List available prompts in the server
# ============================================================
# It advertises reusable prompt templates that the MCP server provides to clients.
# This is prompt discovery
@server.list_prompts() # register this function as the prompt-list endpoint, mcp clients call this to discover available prompts/templates they can use
# This function tells clients: “Here are the prompt templates I support, and how to parameterize them.”
# user types intent like "I want a deep analysis of the document focusing on methodology"
# llm understands that intent and decides to use the deep_analysis prompt template
# client maps that to the deep_analysis prompt with argument query="methodology", server generates the full prompt, sends to llm
# {
  #"name": "deep_analysis",
  #"arguments": { "query": "methodology" }
# }
async def handle_list_prompts() -> list[Prompt]:
    """List available prompts from the server"""
    return [
        Prompt(
            name="deep_analysis",
            description="Perform deep analysis on document sections",
            arguments=[
                PromptArgument(
                    name="query",
                    description="What aspect to analyze (e.g., 'main themes', 'methodology')",
                    required=True
                )
            ]
        ),
        Prompt(
            name="extract_key_information",
            description="Extract specific types of information from document sections",
            arguments=[
                PromptArgument(
                    name="info_type",
                    description="Type of information to extract (definitions, people, statistics, processes, arguments)",
                    required=True
                )
            ]
        )
    ]

# ============================================================
# Handle prompt execution requests
# ============================================================
# It returns a fully-structured prompt (system + user messages) on demand, based on a prompt name and arguments chosen by the LLM/client.
# This is prompt materialization.

@server.get_prompt()
# name → which prompt template (deep_analysis, extract_key_information)
# arguments → parameters supplied by the client/LLM
# Returns GetPromptResult → standardized MCP prompt payload
async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    """Generate a prompt based on the requested type"""
    # Simple argument handling
    if arguments is None:
        arguments = {}
        
    if name == "deep_analysis":
        # Get query with a fallback default
        query = arguments.get("query", "main themes")
        
        # this object defines how the llm should perform deep analysis
        return GetPromptResult(
            description=f"Deep analysis focusing on {query}",
            messages=[
                PromptMessage(
                    role="assistant", 
                    content=TextContent(
                        type="text",
                        text="I am a document analysis expert specializing in identifying key themes, arguments, and evidence in academic and technical documents."
                    )
                ),
                PromptMessage(
                    role="user", 
                    content=TextContent(
                        type="text",
                        text=f"""Please perform a deep analysis of the document section provided in the conversation, focusing on {query}.

Include in your analysis:
- Main themes and arguments presented
- Key evidence and supporting details
- Logical structure and flow of information
- Implicit assumptions made in the text
- Strengths and weaknesses of the arguments
- Connections to broader context if applicable

Format your analysis in a well-structured manner with clear headings and concise explanations.
"""
                    )
                )
            ]
        )
    
    elif name == "extract_key_information":
        # Get info_type with a fallback default
        info_type = arguments.get("info_type", "key information")
        
        return GetPromptResult(
            description=f"Extracting all mentions of {info_type} from document",
            messages=[
                PromptMessage(
                    role="assistant", 
                    content=TextContent(
                        type="text",
                        text="I am a precise information extraction specialist with expertise in technical documents."
                    )
                ),
                PromptMessage(
                    role="user", 
                    content=TextContent(
                        type="text",
                        text=f"""Based on the document section provided in the conversation, please extract all mentions of {info_type}.

Format your response as a structured list with:
1. Clear headers for each extracted element
2. Direct quotes or references when applicable
3. Brief explanations of significance where helpful
4. Page or section references if available

Be comprehensive but focus on quality over quantity. If no mention of the requested info_type is found, just return "No mentions of {info_type} found."
"""
                    )
                )
            ]
        )
    
    else:
        raise ValueError(f"Unknown prompt: {name}")


# ============================================================
# Run the MCP server using stdin/stdout streams
# ============================================================  
# It starts the MCP server and connects it to an MCP client via stdin/stdout.
# This block wires your MCP server into STDIO and keeps it running, ready to handle tool, resource, and prompt requests from any MCP client.
async def main():
    """Run the MCP server using stdin/stdout streams"""
    # Get the distribution info for versioning
    # Version discovery (optional but professional)
    # Attempts to read package version from installed metadata
    # Useful if:server is installed as a package, you want versioned capabilities
    try:
        dist = metadata.distribution("document-search-mcp")
        version = dist.version
    except:
        version = "0.1.0"
    # Opens stdin/stdout as async streams
    # Enables:MCP client ⇄ server communication
    # no HTTP server
    # no ports, no network exposure
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        # Run the MCP server
        # Starts the MCP protocol loop
        # register tools, resources, prompts defined above
        # Handles all incoming requests from the client
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="document-search-mcp",
                server_version=version,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

# ============================================================
# Main Function
# ============================================================
# Allows: python mcp_server.py
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())