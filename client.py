# ============================================================
# Dependencies
# ============================================================

# This client is a command-line interface (CLI) chat app that
# spawns your mcp server over stdio
# discovers tools, resources, and prompts
# sends user queries to openAI
# if openAI request tool calls, the client execute them via the MCP server
# sends tool results back to openAI and gets final response


# async runtime (MCP stdio + OpenAI loop)
import asyncio
# parse tool call arguments (openai sends json strings)
import json
# debug and status log
import logging
# read env vars
import os
# read server script path from command line
import sys
# type hints
from typing import Optional, List, Dict, Any
# cleanly manage multiple async context managers (stdio transport, client session)
from contextlib import AsyncExitStack

# MCP client libraries

# ClientSession - main MCP client session for tool/resource/prompt calls
# StdioServerParameters - parameters to launch MCP server over stdio
# stdio_client - helper to create stdio transport for MCP client (opens stdin/stdout pipes to server)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# OpenAI client library -> call chat.completions.create() and use tool calling
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# Logging
# ============================================================
# enables normal logs for client operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("document-search-client")

# Disable OpenAI and httpx loggers, so terminal stays readable
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ============================================================
# Environment Variables
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# warns if OPENAI_API_KEY is not set
# client can still run and connect to MCP server, but cannot process queries with AI
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set in environment variables")
    logger.warning("The client will not be able to process queries with AI")

# ============================================================
# MCP Client
# ============================================================
class MCPClient:
    # sets up everything needed for the client
    def __init__(self, debug=False):
        """Initialize the MCP client.
        
        Args:
            debug: Whether to enable debug logging
        """
        # ============================================================
        # Initialize session and client objects
        # ============================================================
        
        # will hold mcp session once connected
        self.session: Optional[ClientSession] = None
        # ensures server process pipes + session are cleaned up properly on exit
        self.exit_stack = AsyncExitStack()
        self.debug = debug
        
        # ============================================================
        # Message history tracking
        # ============================================================
        
        # stores user/assistant/tool/system messages for context
        self.message_history = []
        
        # ============================================================
        # Main System Prompt
        # ============================================================

        # instruction sent to openai for behavior guidance
        self.system_prompt = "You are a helpful RAG AI assistant named 'RAG-AI-MCP' that can answer questions about the provided documents or query the attached database for more information."

        # ============================================================
        # Initialize OpenAI Client
        # ============================================================
        try:
            # Initialize OpenAI client if API key is provided, reused across requests
            self.openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
            if not self.openai:
                logger.warning("OpenAI client not initialized - missing API key")
        
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            self.openai = None
        
        # ============================================================
        # Server connection info
        # ============================================================
       
        # populated after connecting to server
        # server name for display
        self.available_tools = []
        self.available_resources = []
        self.available_prompts = []
        self.server_name = None

    # ============================================================
    # Connect to MCP Server
    # ============================================================

    # spawn server + open pipes + create session + discover capabilities.
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        if self.debug:
            logger.info(f"Connecting to server at {server_script_path}")
            
        # Check for existing Python script
        # validates server script path is a .py file
        is_python = server_script_path.endswith('.py')
        if not (is_python):
            raise ValueError("Server script must be a .py file")

        # Initialize server parameters
        # Launch the server process over stdio
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )

        # Initialize stdio transport
        try:
            # stdio_client(server_params) → launches server and returns read/write streams
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            # ClientSession(self.stdio, self.write) → MCP session over those streams.
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            
            # Initialize the session
            # MCP handshake; gets server name/version
            init_result = await self.session.initialize()
            self.server_name = init_result.serverInfo.name
            
            if self.debug:
                logger.info(f"Connected to server: {self.server_name} v{init_result.serverInfo.version}")
            
            # Cache available tools, resources, and prompts
            # calls list_tools/resources/prompts and caches them.
            await self.refresh_capabilities()
            
            # return true or false based on success of connection
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    # ============================================================
    # Refresh Server Capabilities
    # ============================================================
    
    # Calls discovery endpoints: list_tools, list_resources, list_prompts
    # the LLM needs these schemas; also CLI commands use them.
    async def refresh_capabilities(self):
        """Refresh the client's knowledge of server capabilities"""
        if not self.session:
            raise ValueError("Not connected to server")
            
        # Get available tools
        tools_response = await self.session.list_tools()
        self.available_tools = tools_response.tools
        
        # Get available resources
        resources_response = await self.session.list_resources()
        self.available_resources = resources_response.resources
        
        # Get available prompts
        prompts_response = await self.session.list_prompts()
        self.available_prompts = prompts_response.prompts
        
        if self.debug:
            logger.info(f"Server capabilities refreshed:")
            logger.info(f"- Tools: {len(self.available_tools)}")
            logger.info(f"- Resources: {len(self.available_resources)}")
            logger.info(f"- Prompts: {len(self.available_prompts)}")

    # ============================================================
    # Handling Message History Helper Function
    # ============================================================
    
    # History Helper
    # stores messages with role(user,assistant,system,tool), content, timestamp, metadata
    # lets you rebuild OpenAI “messages” later and preserve tool call ordering
    async def add_to_history(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the history
        
        Args:
            role: The role of the message sender (user, assistant, system, resource)
            content: The message content
            metadata: Optional metadata about the message
        """
        
        # Format message
        message = {
            "role": role,
            "content": content,
            "timestamp": asyncio.get_event_loop().time(),
            "metadata": metadata or {}
        }

        # Add message to history
        self.message_history.append(message)
        
        if self.debug:
            logger.info(f"Added message to history: {role} - {content[:100]}...")

    # ============================================================
    # List Available Resources from the MCP Server
    # ============================================================
    
    # Resource Helpers
    # just re-fetches resources from server and caches them
    async def list_resources(self):
        """List available resources from the MCP server"""
        if not self.session:
            raise ValueError("Not connected to server")
            
        response = await self.session.list_resources()
        self.available_resources = response.resources
        
        if self.debug:
            resource_uris = [res.uri for res in self.available_resources]
            logger.info(f"Available resources: {resource_uris}")
        
        return self.available_resources

    # ============================================================
    # Read Content from a Resource and Add to Message History
    # ============================================================
    
    # Resource Helpers
    # reads resource content from server and adds to message history
    # Calls self.session.read_resource(uri) to get raw text from server
    # Adds that text into history as a user message (so the LLM can use it as context)
    # Returns the content (or error string)
    async def read_resource(self, uri: str):
        """Read content from a specific resource
        
        Args:
            uri: The URI of the resource to read
        
        Returns:
            The content of the resource as a string
        """
            
        if self.debug:
            logger.info(f"Reading resource: {uri}")
            
        try:
            # Read resource content
            result = await self.session.read_resource(uri)
            
            # Check if resource content is found
            if not result:
                content = "No content found for this resource."
            else:
                content = result if isinstance(result, str) else str(result)
            
            # Add resource content to history as a user message
            resource_message = f"Resource content from {uri}:\n\n{content}"
            await self.add_to_history("user", resource_message, {"resource_uri": uri, "is_resource": True})
            
            return content
        
        except Exception as e:
            error_msg = f"Error reading resource {uri}: {str(e)}"
            logger.error(error_msg)
            await self.add_to_history("user", error_msg, {"uri": uri, "error": True})
            return error_msg

    # ============================================================
    # List Available Prompts from the MCP Server
    # ============================================================
    
    # Prompt Helpers
    # Fetches prompt descriptors from server.
    async def list_prompts(self):
        """List available prompts from the MCP server"""
        
        # Get available prompts
        response = await self.session.list_prompts()
        self.available_prompts = response.prompts
        
        if self.debug:
            prompt_names = [prompt.name for prompt in self.available_prompts]
            logger.info(f"Available prompts: {prompt_names}")
        
        return self.available_prompts

    # ============================================================
    # Get a Specific Prompt with Arguments
    # ============================================================
    
    # Prompt Helpers
    # Fetches actual prompt template (GetPromptResult) from server
    async def get_prompt(self, name: str, arguments: dict = None):
        """Get a specific prompt with arguments
        
        Args:
            name: The name of the prompt
            arguments: Optional arguments to pass to the prompt
            
        Returns:
            The prompt result
        """
            
        if self.debug:
            logger.info(f"Getting prompt: {name} with arguments: {arguments}")
            
        try:
            # Get the prompt
            prompt_result = await self.session.get_prompt(name, arguments)
            return prompt_result
        except Exception as e:
            error_msg = f"Error getting prompt {name}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # ============================================================
    # Process a Query using OpenAI and Available Tools
    # ============================================================
    
    # Main RAG Logic
    # This is the important part: OpenAI tool loop + MCP execution.
    # This function builds a valid OpenAI chat history, lets OpenAI decide tool calls, executes those tools on the MCP server, 
    # then calls OpenAI again to generate the final response.
    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available tools
        
        Args:
            query: The query to process
            
        Returns:
            The response from the AI after processing the query
        """
            
        # Add user query to history
        # Stores the user question in message_history so future turns have context.
        await self.add_to_history("user", query)
        
        # Convert message history to OpenAI format
        messages = []
        
        # Always include the current system prompt first
        # OpenAI chat needs a messages list. System prompt sets assistant behavior every call.
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # We need to properly maintain the tool call sequence
        # This means ensuring every 'tool' message follows an 'assistant' message with tool_calls
        # OpenAI requires this ordering:
        # assistant message with tool_calls
        # tool messages with matching tool_call_id
        # then another assistant message
        # So the client rebuilds history carefully to avoid OpenAI API errors.
        
        # holds an assistant message that declared tool calls
        assistant_with_tool_calls = None
        # holds tool results to attach right after it
        pending_tool_responses = []
        
        # Track message indices to help with debugging
        # Rebuild conversation from self.message_history
        # Iterate through all previous messages in your internal format.
        for i, msg in enumerate(self.message_history):
            # Handle different message types
            # 1) User messages
            if msg['role'] == 'user':
                # First flush any pending tool responses if needed
                # Before adding a new user message, flush any pending assistant+tool results pair (so ordering stays valid).
	            # Then add the user message to OpenAI.
                if assistant_with_tool_calls and pending_tool_responses:
                    messages.append(assistant_with_tool_calls)
                    messages.extend(pending_tool_responses)
                    assistant_with_tool_calls = None
                    pending_tool_responses = []
                
                # Then add the user message
                messages.append({
                    "role": "user",
                    "content": msg['content']
                })
            
            # 2) Assistant messages
            elif msg['role'] == 'assistant':
                # Check if this is an assistant message with tool calls
                # Reads metadata you stored earlier (like tool_calls).
                metadata = msg.get('metadata', {})
                # Assistant message with tool calls
                if metadata.get('has_tool_calls', False):
                    # If we already have a pending assistant with tool calls, flush it
                    # If this assistant initiated tool calls, don’t immediately append it.
                    # Store it until you gather the matching tool results, then append assistant+tools together.
                    if assistant_with_tool_calls:
                        messages.append(assistant_with_tool_calls)
                        messages.extend(pending_tool_responses)
                        pending_tool_responses = []
                    
                    # Store this assistant message for later (until we collect all tool responses)
                    assistant_with_tool_calls = {
                        "role": "assistant",
                        "content": msg['content'],
                        "tool_calls": metadata.get('tool_calls', [])
                    }
                # Regular assistant message (no tool calls)
                else:
                    # Regular assistant message without tool calls
                    # First flush any pending tool calls
                    # Before adding normal assistant text, flush any pending tool-call sequence.
                    # Then add assistant message normally.
                    if assistant_with_tool_calls:
                        messages.append(assistant_with_tool_calls)
                        messages.extend(pending_tool_responses)
                        assistant_with_tool_calls = None
                        pending_tool_responses = []
                    
                    # Then add the regular assistant message
                    messages.append({
                        "role": "assistant",
                        "content": msg['content']
                    })
            
            # 3) Tool messages
            elif msg['role'] == 'tool' and 'tool_call_id' in msg.get('metadata', {}):
                # Collect tool responses
                # Tool results must come after the assistant message that declared them.
                # So it buffers tool messages until the right assistant is appended.
                if assistant_with_tool_calls:  # Only add if we have a preceding assistant message with tool calls
                    pending_tool_responses.append({
                        "role": "tool",
                        "tool_call_id": msg['metadata']['tool_call_id'],
                        "content": msg['content']
                    })
            
            # 4) System messages
            # Adds any extra system messages directly.
            elif msg['role'] == 'system':
                # System messages can be added directly
                messages.append({
                    "role": "system",
                    "content": msg['content']
                })
        
        # Flush any remaining pending tool calls at the end
        # E) Final flush (if loop ended mid tool-sequence)
        # Ensures no buffered tool-call sequence is left out.
        if assistant_with_tool_calls:
            messages.append(assistant_with_tool_calls)
            messages.extend(pending_tool_responses)
        
        # F) Debug print (optional)
        # Logs the final OpenAI messages list for troubleshooting.
        if self.debug:
            logger.info(f"Prepared {len(messages)} messages for OpenAI")
            for i, msg in enumerate(messages):
                role = msg['role']
                has_tool_calls = 'tool_calls' in msg
                preview = msg['content'][:50] + "..." if msg['content'] else ""
                logger.info(f"Message {i}: {role} {'with tool_calls' if has_tool_calls else ''} - {preview}")
        
        # Make sure we have the latest tools
        # G) Ensure tool list is loaded
        # If tools weren’t cached yet, fetch them from the MCP server.
        if not self.available_tools:
            await self.refresh_capabilities()

        # Format tools for OpenAI
        # H) Convert MCP tool schema → OpenAI tool schema
        # OpenAI expects tool definitions in this JSON structure. MCP provides Tool objects → you map them.
        available_tools = [{ 
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in self.available_tools]

        if self.debug:
            tool_names = [tool["function"]["name"] for tool in available_tools]
            logger.info(f"Available tools for query: {tool_names}")
            logger.info(f"Sending {len(messages)} messages to OpenAI")

        # Initial OpenAI API call
        # I) First OpenAI call (decide answer vs tool calls)
        # Sends user + history + tool schemas.
        # auto lets OpenAI choose whether to call a tool.
        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=available_tools,
                tool_choice="auto"
            )
        # If it fails:
        # Stores error in history and returns it.
        except Exception as e:
            error_msg = f"Error calling OpenAI API: {str(e)}"
            logger.error(error_msg)
            await self.add_to_history("assistant", error_msg, {"error": True})
            return error_msg

        # Process response and handle tool calls
        tool_results = []
        final_text = []
        
        # J) Read assistant reply + store it
        # Gets the model’s first response (may be empty if it only wants tools).
        assistant_message = response.choices[0].message
        initial_response = assistant_message.content or ""
        
        # Add initial assistant response to history with metadata about tool calls
        # Saves assistant message to history, including tool call info so future turns can reconstruct ordering.
        tool_calls_metadata = {}
        if assistant_message.tool_calls:
            tool_calls_metadata = {
                "has_tool_calls": True,
                "tool_calls": assistant_message.tool_calls
            }
        
        await self.add_to_history("assistant", initial_response, tool_calls_metadata)
        final_text.append(initial_response)
        
        # Check if tool calls are present
        # K) If OpenAI requested tool calls: execute them via MCP
        # OpenAI requires you to add the assistant tool_call message into the conversation before tool results.
        if assistant_message.tool_calls:
            if self.debug:
                logger.info(f"Tool calls requested: {len(assistant_message.tool_calls)}")
            
            # Add the assistant's message to the conversation
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls
            })
            
            # Process each tool call
            # For each tool call:
            # Extract tool name + args OpenAI requested.
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                
                # Convert json string to dict if needed
                # Parse args if JSON string:
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments as JSON: {tool_args}")
                        tool_args = {}
                
                if self.debug:
                    logger.info(f"Executing tool: {tool_name}")
                    logger.info(f"Arguments: {tool_args}")
                
                # Execute tool call on the server
                try:
                    # Call MCP server tool:
                    # This triggers your server’s @server.call_tool() handler.
                    result = await self.session.call_tool(tool_name, tool_args)
                    
                    # Extract MCP result content:
                    # Your server returns TextContent, so you read .text.
                    tool_content = result.content if hasattr(result, 'content') else str(result)
                    tool_results.append({"call": tool_name, "result": tool_content[0].text})
                    final_text.append(f"\n[Calling tool {tool_name} with args {tool_args}]")
                    
                    if self.debug:
                        result_preview = tool_content[0].text[:200] + "..." if len(tool_content[0].text) > 200 else tool_content[0].text
                        logger.info(f"Tool result preview: {result_preview}")
                    
                    # Add the tool result to the conversation
                    # Add tool output back to OpenAI messages (MANDATORY):
                    # This is the OpenAI tool-result format. tool_call_id must match.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content[0].text
                    })
                    # Also store tool output in history:
                    # If tool execution fails, you still append a "tool" error message so OpenAI can continue:
                    await self.add_to_history("tool", tool_content[0].text, {"tool": tool_name, "args": tool_args, "tool_call_id": tool_call.id})
                
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    logger.error(error_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg
                    })
                    await self.add_to_history("tool", error_msg, {"tool": tool_name, "error": True, "tool_call_id": tool_call.id})
                    final_text.append(f"\n[Error executing tool {tool_name}: {str(e)}]")
            
            if self.debug:
                logger.info("Getting final response from OpenAI with tool results")
            
            # Get a new response from OpenAI with the tool results
            try:
                second_response = self.openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                
                response_content = second_response.choices[0].message.content or ""
                await self.add_to_history("assistant", response_content)
                final_text.append("\n" + response_content)

            except Exception as e:
                error_msg = f"Error getting final response from OpenAI: {str(e)}"
                logger.error(error_msg)
                await self.add_to_history("assistant", error_msg, {"error": True})
                final_text.append(f"\n[Error: {error_msg}]")

        return "\n".join(final_text)

    # ============================================================
    # Main Chat Loop
    # ============================================================

    # 1; For normal queries (no slash command)
    # This path uses process_query(), which:
	# writes to message_history (persistent log)
	# builds messages (OpenAI payload)
	# does OpenAI tool loop + MCP calls

    # So that’s the “full” pipeline.
    

    # 2; For /tools, /resources, /prompts
    # These are direct MCP discovery calls + printing.
	# /tools → prints self.available_tools (already cached from refresh)
	# /resources → calls list_resources() then prints
	# /prompts → calls list_prompts() then prints
    # No OpenAI call. No messages building. No tool loop.
    # (They may update the cached lists, but they’re not doing reasoning.)


    # 3) For /resource <uri>
    # This is still direct MCP, but with one extra thing:
	# it calls read_resource(uri)
	# and read_resource() adds the resource text into message_history as a "user" message
    # So later, when you ask a normal query, OpenAI will “see” that resource content (because process_query() rebuilds messages from history).

    # 4) For /prompt ...
    # This is the special case 
    # Here you are doing something similar to process_query(), but simpler:
	# It calls MCP get_prompt() → gets a prompt template (prompt_result.messages)
	# Then it builds a new list called openai_messages (this is like messages, but separate variable name)
	# It first pulls a few recent messages from message_history for context
	# Then appends the prompt template messages
	# Then it calls OpenAI once with openai_messages
    # After it gets the response:
	# it adds the prompt messages + assistant response into message_history



    async def chat_loop(self):
        """Welcome to the RAG-AI-MCP Client!"""
        print(f"\n{'='*50}")
        print(f"RAG-AI-MCP Client Connected to: {self.server_name}")
        print(f"{'='*50}")
        print("Type your queries or use these commands:")
        print("  /debug - Toggle debug mode")
        print("  /refresh - Refresh server capabilities")
        print("  /resources - List available resources")
        print("  /resource <uri> - Read a specific resource")
        print("  /prompts - List available prompts")
        print("  /prompt <name> <argument> - Use a specific prompt with a string as the argument")
        print("  /tools - List available tools")
        print("  /quit - Exit the client")
        print(f"{'='*50}")
        
        # Main chat loop
        while True:
            try:
                # Get user query
                query = input("\nQuery: ").strip()
                
                # Handle commands
                if query.lower() == '/quit':
                    break

                # Toggle debug mode
                elif query.lower() == '/debug':
                    self.debug = not self.debug
                    print(f"\nDebug mode {'enabled' if self.debug else 'disabled'}")
                    continue

                # Refresh server capabilities
                elif query.lower() == '/refresh':
                    await self.refresh_capabilities()
                    print("\nServer capabilities refreshed")
                    continue

                # List available resources
                elif query.lower() == '/resources':
                    resources = await self.list_resources()
                    print("\nAvailable Resources:")
                    for res in resources:
                        print(f"  - {res.uri}")
                        if res.description:
                            print(f"    {res.description}")
                    continue

                # Read content from a resource
                elif query.lower().startswith('/resource '):
                    uri = query[10:].strip()
                    print(f"\nFetching resource: {uri}")
                    content = await self.read_resource(uri)
                    print(f"\nResource Content ({uri}):")
                    print("-----------------------------------")
                    # Print first 500 chars with option to see more
                    if len(content) > 500:
                        print(content[:500] + "...")
                        print("(Resource content truncated for display purposes but full content is included in message history)")
                    else:
                        print(content)
                    continue

                # List available prompts
                elif query.lower() == '/prompts':
                    prompts = await self.list_prompts()
                    print("\nAvailable Prompts:")
                    for prompt in prompts:
                        print(f"  - {prompt.name}")
                        if prompt.description:
                            print(f"    {prompt.description}")
                        if prompt.arguments:
                            print(f"    Arguments: {', '.join(arg.name for arg in prompt.arguments)}")
                    continue

                # Run a specific prompt with arguments
                elif query.lower().startswith('/prompt '):
                    
                    # Parse: /prompt name sentence of arguments
                    parts = query[8:].strip().split(maxsplit=1)
                    if not parts:
                        print("Error: Prompt name required")
                        continue
                    
                    name = parts[0]
                    arguments = {}
                    
                    # If there are arguments (anything after the prompt name)
                    if len(parts) > 1:
                        arg_text = parts[1]
                        
                        # Get the prompt to check its expected arguments
                        prompt_info = None
                        for prompt in self.available_prompts:
                            if prompt.name == name:
                                prompt_info = prompt
                                break
                                
                        if prompt_info and prompt_info.arguments and len(prompt_info.arguments) > 0:
                            # Use the first argument name as the key for the entire sentence
                            arguments[prompt_info.arguments[0].name] = arg_text
                        else:
                            # Default to using "text" as the argument name if no prompt info available
                            arguments["text"] = arg_text
                    
                    print(f"\nGetting prompt template: {name}")
                    prompt_result = await self.get_prompt(name, arguments)
                    
                    # Process the prompt with OpenAI and add to conversation
                    if not self.openai:
                        print("Error: OpenAI client not initialized. Cannot process prompt.")
                        continue
                        
                    messages = prompt_result.messages
                    
                    # Convert messages to OpenAI format and include relevant history
                    openai_messages = []
                    
                    # First add the last few user messages to provide document context
                    # (up to 5 recent messages but skip system messages and error messages)
                    recent_messages = []
                    for msg in reversed(self.message_history[-10:]):
                        if msg['role'] in ['user', 'assistant'] and len(recent_messages) < 5:
                            recent_messages.append({
                                "role": msg['role'],
                                "content": msg['content']
                            })
                    
                    # Add recent messages in correct order (oldest first)
                    openai_messages.extend(reversed(recent_messages))
                    
                    # Then add the prompt messages
                    for msg in messages:
                        content = msg.content.text if hasattr(msg.content, 'text') else str(msg.content)
                        openai_messages.append({
                            "role": msg.role,
                            "content": content
                        })
                    
                    print("Processing prompt...")

                    try:
                        response = self.openai.chat.completions.create(
                            model="gpt-4o",
                            messages=openai_messages
                        )
                        
                        response_content = response.choices[0].message.content
                        # Add the prompt and response to conversation history
                        for msg in messages:
                            content = msg.content.text if hasattr(msg.content, 'text') else str(msg.content)
                            await self.add_to_history(msg.role, content)
                        
                        await self.add_to_history("assistant", response_content)
                        
                        print("\nResponse:")
                        print(response_content)
                    
                    except Exception as e:
                        error_msg = f"\nError processing prompt with OpenAI: {str(e)}"
                        print(error_msg)
                    continue
                
                # List available tools
                elif query.lower() == '/tools':
                    print("\nAvailable Tools:")
                    for tool in self.available_tools:
                        print(f"  - {tool.name}")
                        if tool.description:
                            print(f"    {tool.description}")
                    continue
                    
                # Process regular queries
                print("\nProcessing query...")
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
    
    # ============================================================
    # Resource Cleanup
    # ============================================================  
    async def cleanup(self):
        """Clean up resources"""
        if self.debug:
            logger.info("Cleaning up client resources")
        await self.exit_stack.aclose()

# ============================================================
# Main Function
# ============================================================
async def main():
    """Run the MCP client"""

    # Check for server script path
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
    
    # Initialize client
    server_script = sys.argv[1]
    client = MCPClient()
    
    # Connect to server
    try:
        connected = await client.connect_to_server(server_script)
        if not connected:
            print(f"Failed to connect to server at {server_script}")
            sys.exit(1)
            
        # Start chat loop
        await client.chat_loop()
    
    # Handle other exceptions
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    # Cleanup resources
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())