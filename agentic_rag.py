from typing import TypedDict, Annotated, List, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from typing_extensions import List, TypedDict

from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import operator
from langgraph.checkpoint.memory import MemorySaver
import uuid

import logging

# ====================== LOGGING SETUP ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger("langgraph_agent")# Sample technical documentation
docs_content = [
    {
        "content": """REST API Authentication
        
The API uses JWT (JSON Web Tokens) for authentication. To authenticate:

1. Send POST request to /api/v1/auth/login with credentials
2. Receive JWT token in response
3. Include token in Authorization header: Bearer <token>
4. Tokens expire after 24 hours

Example:
POST /api/v1/auth/login
{"username": "user", "password": "pass"}

Response:
{"token": "eyJhbGc...", "expires_in": 86400}""",
        "metadata": {"source": "auth_guide", "section": "authentication"}
    },
    {
        "content": """Rate Limiting

API requests are rate limited to prevent abuse:

- 100 requests per minute per API key
- 1000 requests per hour per API key
- Rate limit headers included in responses:
  X-RateLimit-Limit: Maximum requests allowed
  X-RateLimit-Remaining: Requests remaining
  X-RateLimit-Reset: Time when limit resets

Exceeding rate limits returns HTTP 429 (Too Many Requests).""",
        "metadata": {"source": "rate_limits", "section": "limits"}
    },
    {
        "content": """WebSocket Connections

Real-time data streaming via WebSocket:

- Endpoint: wss://api.example.com/v1/stream
- Requires authentication via query parameter: ?token=<jwt>
- Sends JSON messages with event types
- Automatic reconnection on disconnect
- Ping/pong heartbeat every 30 seconds

Connection example:
const ws = new WebSocket('wss://api.example.com/v1/stream?token=<jwt>');""",
        "metadata": {"source": "websocket_guide", "section": "realtime"}
    },
    {
        "content": """Error Handling

Standard HTTP error codes:

400 Bad Request - Invalid request format or parameters
401 Unauthorized - Missing or invalid authentication
403 Forbidden - Valid auth but insufficient permissions
404 Not Found - Endpoint or resource doesn't exist
429 Too Many Requests - Rate limit exceeded
500 Internal Server Error - Server-side error
503 Service Unavailable - Service temporarily down

Error responses include:
{"error": "error_code", "message": "description", "details": {}}""",
        "metadata": {"source": "error_guide", "section": "errors"}
    },
    {
        "content": """Data Models

User object structure:
{
  "id": "string (UUID)",
  "username": "string",
  "email": "string",
  "created_at": "ISO 8601 timestamp",
  "roles": ["string array"],
  "status": "active|suspended|deleted"
}

Project object structure:
{
  "id": "string (UUID)",
  "name": "string",
  "owner_id": "string (user UUID)",
  "created_at": "ISO 8601 timestamp",
  "settings": {}
}""",
        "metadata": {"source": "data_models", "section": "schemas"}
    },
    {
        "content": """API Endpoints Reference

GET /api/v1/users - List all users (paginated)
GET /api/v1/users/{id} - Get specific user
POST /api/v1/users - Create new user
PUT /api/v1/users/{id} - Update user
DELETE /api/v1/users/{id} - Delete user

GET /api/v1/projects - List projects
GET /api/v1/projects/{id} - Get project details
POST /api/v1/projects - Create project
PUT /api/v1/projects/{id} - Update project

All endpoints require authentication except /auth/login.""",
        "metadata": {"source": "endpoints", "section": "reference"}
    }
]

# Convert to Document objects
documents = [
    Document(page_content=doc["content"], metadata=doc["metadata"])
    for doc in docs_content
]

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = InMemoryVectorStore(embeddings)
_ = vectorstore.add_documents(documents=documents)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(f"Indexed {len(documents)} documents")

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_docs_with_sources(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" 
        for doc in docs
        )

class GenerationGrade(BaseModel):
    """Schema for grading the generated answer."""
    supported: str = Field(..., description="Is the generation grounded in the documents? 'yes' or 'no'")
    useful: str = Field(..., description="Does the generation answer the question? 'yes' or 'no'")

class GraphState(TypedDict):
    # Input
    question: str                    # This may be rewritten
    original_question: str           # ← Add this (never overwritten)
    
    # RAG pipeline data
    documents: List[Document]
    generation: str
    generation_grade: GenerationGrade | None
    
    # Control variables
    rewrite_count: int
    generation_retries: int
    
    # === INTERNAL MEMORY (seen by the agent) ===
    messages: Annotated[List[BaseMessage], operator.add]
    
    # === CLEAN HISTORY FOR UI ===
    chat_history: Annotated[List[dict], operator.add]   # Only what user sees

# Helpers
def get_retrieval_grader():
    llm = ChatOllama(model='gemma4:31b', format="json", temperature=0)
    prompt = PromptTemplate.from_template(
        """You are a grader assessing relevance of a retrieved document to a user question. \n
Here is the retrieved document: \n\n {document} \n\n
Here is the user question: {question} \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
Provide the binary score as a JSON with a single key 'score' and no premable or explanation."""
    )
    retrieval_grader = prompt | llm | JsonOutputParser()
    return retrieval_grader

def get_hallucination_grader():
    llm = ChatOllama(model='gemma4:31b', format="json", temperature=0)
    prompt = PromptTemplate.from_template(
        """You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
Here are the facts:
\n ------- \n
{documents} 
\n ------- \n
Here is the answer: {generation} \n
Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in / supported by a set of facts. \n
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""
    )
    hallucination_grader = prompt | llm | JsonOutputParser()
    return hallucination_grader

def get_answer_grader():
    llm = ChatOllama(model='gemma4:31b', format="json", temperature=0)
    prompt = PromptTemplate.from_template(
        """You are a grader assessing whether an answer is useful to resolve a question. \n 
Here is the answer:
\n ------- \n
{generation} 
\n ------- \n
Here is the question: {question} \n
Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""
    )
    answer_grader = prompt | llm | JsonOutputParser()
    return answer_grader

def get_question_rewriter():
    llm = ChatOllama(model='gemma4:31b', temperature=0)
    rewrite_prompt = PromptTemplate.from_template(
        """You're a question re-writer that converts an input question to a better version that is optimized \
for vectorstore retrieval. Look at the initial and formulate an improved question. \n
Here is the initial question: \n {question} \n
Improved question with no preamble: \n"""
    )
    question_rewriter = rewrite_prompt | llm | StrOutputParser()
    return question_rewriter

def get_question_router():  #########
    llm = ChatOllama(model='gemma4:31b', format="json", temperature=0)
    prompt = PromptTemplate.from_template(
        """You are an expert at routing a user question to a vectorstore or web search. \n
        Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. \n
        You do not need to be stringent with the keywords in the question related to these topics. \n
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
        Return the a JSON with a single key 'datasource' and no premable or explanation. \n
        Question to route: {question}"""
    )
    question_router = prompt | llm | JsonOutputParser()
    return question_router

# Nodes
def retrieve_node(state: GraphState) -> dict:
    """Retrieve documents from vector store."""
    question = state["question"]
    logger.info(f"RETRIEVE: Question = '{question}'")

    documents = retriever.invoke(question)

    logger.info(f"RETRIEVE: Retrieved {len(documents)} documents")
    for i, doc in enumerate(documents[:3]):   # log first 3
        logger.debug(f"   Doc {i+1} | Source: {doc.metadata.get('source', 'N/A')[:80]}")

    return {
        "documents": documents,
        "rewrite_count": state.get("rewrite_count", 0)
    }

def generate_node(state: GraphState) -> dict:
    question = state["question"]                    # rewritten version (for retrieval)
    original_question = state.get("original_question", question)
    documents = state.get("documents", [])
    retries = state.get("generation_retries", 0)
    
    logger.info(f"GENERATE: Attempt {retries + 1} | Docs count: {len(documents)}")

    if not documents:
        generation = "I don't have enough information to answer this question."
    else:
        llm = ChatOllama(model='gemma4:31b', temperature=0)

        qa_system_prompt = """You are a helpful assistant. Answer using only the following context. 
Be specific and cite sources. If you don't know, say 'I don't know'.\n\nContext:\n{context}"""

        qa_prompt_template = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="messages"),   # Full history
            ("human", "{question}")
        ])

        question_answer_chain = create_stuff_documents_chain(
            llm=llm, 
            prompt=qa_prompt_template
        )

        generation = question_answer_chain.invoke({
            "context": documents,
            "question": question,
            "messages": state.get("messages", [])
        })

    # === RETURN UPDATED STATE ===
    return {
        "generation": generation,
        "generation_retries": retries + 1,
        
        # Internal messages (agent memory)
        "messages": [
            HumanMessage(content=question),
            AIMessage(
                content=generation,
                additional_kwargs={
                    "produced_by": "generate_node",
                    "documents_count": len(documents),
                    "final_answer": True
                }
            )
        ],
        
        # Clean history for UI / Database
        "chat_history": [
            {"role": "user", "content": original_question},   # ← Use original!
            {"role": "assistant", "content": generation}
        ]
    }

def grade_generation(state: GraphState) -> dict:
    """    Determines whether the generation is grounded in the document and answers question."""
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    logger.info("GRADE_GENERATION: Evaluating answer quality...")

    hallucination_grader = get_hallucination_grader()
    answer_grader = get_answer_grader()

    hallucination_score = hallucination_grader.invoke(
        {"documents": format_docs(documents), "generation": generation}######################
    )
    supported = hallucination_score["score"]

    answer_score = answer_grader.invoke(
        {"question": question, "generation": generation}
    )
    useful = answer_score["score"]

    logger.info(f"GRADE_GENERATION: Supported = {supported} | Useful = {useful}")

    generation_grade = GenerationGrade(supported=supported, useful=useful)

    return {
        "generation_grade": generation_grade
    }

def grade_documents(state: GraphState) -> dict:
    """Score documents for relevance to the question."""
    question = state["question"]
    documents = state["documents"]

    logger.info(f"GRADE_DOCUMENTS: Grading {len(documents)} documents")
    
    retrieval_grader = get_retrieval_grader()
    filtered_docs = []
    for i, doc in enumerate(documents):
        score = retrieval_grader.invoke(
                    {"question": question, "document": doc.page_content}
                )
        grade = score["score"]

        logger.info(f"   Doc {i+1} → Relevance: {grade}")

        if grade == "yes":
            filtered_docs.append(doc)

    logger.info(f"GRADE_DOCUMENTS: Kept {len(filtered_docs)} relevant documents")
    return {"documents": filtered_docs}

def rewrite_query(state: GraphState):
    """Rewrite the query to improve retrieval."""
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    logger.info(f"REWRITE_QUERY: Attempt {rewrite_count + 1} | Original: '{question}'")

    question_rewriter = get_question_rewriter()
    better_question = question_rewriter.invoke({"question": question})

    logger.info(f"REWRITE_QUERY: New question = '{better_question}'")
    
    return {
        "question": better_question,
        "rewrite_count": rewrite_count + 1
    }

def generate_fallback(state: GraphState):
    """
    Final fallback node.
    Handles two cases:
    1. No documents were retrieved.
    2. Retry limit reached with bad generation quality.
    """

    generation = "I don't have information to answer the question."
    return {
        "generation": generation,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=generation)
        ],
        "chat_history": [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": generation}
        ]
    }

def web_search_node(state: GraphState):
    """Search the web when internal docs are insufficient."""
    question = state["question"]
    web_search_tool = TavilySearchResults(max_results=3)
    results = web_search_tool.invoke(question)
    
    docs = [
        Document(page_content=r["content"], metadata={"source": r["url"]})
        for r in results
    ]
    return {"documents": docs}

# Decision functions
# def decide_next_step(state: GraphState):
def decide_to_generate(state: GraphState):
    """Decide next step after grading documents."""
    docs_count = len(state.get("documents", []))
    rewrite_count = state.get("rewrite_count", 0)

    logger.info(f"DECIDE: {docs_count} relevant docs | Rewrites used: {rewrite_count}")

    if docs_count > 0:
        logger.info("DECIDE → Go to GENERATE")
        return "generate"
    elif state["rewrite_count"] < 1:
        logger.info("DECIDE → Go to REWRITE_QUERY")
        return "rewrite"
    else:
        logger.warning("DECIDE → Go to FALLBACK (no good docs after rewrite)")
        return "generate_fallback"

def generation_router(state: GraphState):
    """
    Lightweight router after generation grading.
    """
    generation_grade: GenerationGrade = state.get("generation_grade")
    retries = state.get("generation_retries", 0)
    MAX_RETRIES = 3

    supported = generation_grade.supported
    useful = generation_grade.useful
    is_good_quality = (supported == "yes" and useful == "yes")

    logger.info(f"ROUTER: Quality = {'GOOD' if is_good_quality else 'BAD'} | Retries = {retries}/{MAX_RETRIES}")

    if is_good_quality:
        logger.info("ROUTER → END (Good answer)")
        return "good_quality"               # → END
    elif retries < MAX_RETRIES:
        logger.info("ROUTER → RETRY Generation")
        return "bad_quality"                # → generate (retry)
    else:
        logger.warning("ROUTER → FALLBACK (Max retries reached)")
        return "generation_fallback"        # → generation_fallback
    
    # Graph
    from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
# workflow.add_node("web_search", web_search_node)  # web search
workflow.add_node("retrieve", retrieve_node)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate_node)  # generate
workflow.add_node("grade_generation", grade_generation)
workflow.add_node("rewrite_query", rewrite_query)  # rewrite query
workflow.add_node("generate_fallback", generate_fallback)

# Build graph
workflow.set_entry_point("retrieve")
# workflow.add_conditional_edges(
#     START,
#     route_question,
#     {
#         "web_search": "web_search",
#         "vectorstore": "retrieve",
#     },
# )
# workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "rewrite": "rewrite_query",
        "generate": "generate",
        "generate_fallback": "generate_fallback"
    },
)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", "grade_generation")
workflow.add_conditional_edges(
    "grade_generation",
    generation_router,
    {
        "bad_quality": "generate",
        "good_quality": END,
        "generate_fallback": "generate_fallback"
    },
)
workflow.add_edge("generate_fallback", END)

# Compile
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

thread_id = str(uuid.uuid4())


def chat_with_agent(question: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    input_state = {
        "question": question,
        "original_question": question,
        "rewrite_count": 0,
        "generation_retries": 0,
        "messages": [],        # LangGraph will load previous ones automatically
        "chat_history": []     # Also loaded automatically
    }
    
    result = app.invoke(input_state, config)

    # For development - simple in-memory store
    if 'chat_db' not in globals():
        global chat_db
        chat_db = {}
    
    if thread_id not in chat_db:
        chat_db[thread_id] = []
    
    chat_db[thread_id].extend(result.get("chat_history", []))
    
    # Save clean history to your database for the frontend
    # save_chat_to_db(
    #     thread_id=thread_id,
    #     messages=result.get("chat_history", [])
    # )
    
    return result["generation"]
