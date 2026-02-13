from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool
import os
from datetime import datetime


def create_company_knowledge_base():
    """
    Create a vector store from company documents
    This will be the company's knowledge base
    """

    # Sample company information (replace with your actual company data)
    company_info = """
    # E-Commerce Quality Control Solutions Company
    
    ## Company Overview
    OGE.qc is a leading provider of AI-powered quality control solutions for e-commerce businesses.
    Founded in 2024, we help online retailers ensure product quality through advanced machine learning models.
    
    ## Our Products and Services
    
    ### 1. Price Prediction System
    Our regression-based price prediction model helps retailers optimize product pricing.
    - Analyzes market trends, competitor pricing, and product features
    - Provides price recommendations with confidence intervals
    - Supports multiple product categories: Electronics, Clothing, Home & Garden
    - Accuracy rate: 80%
    
    ### 2. Quality Classification System
    Automated quality grading system for product inspection.
    - Grades products from A to F based on multiple quality metrics
    - Analyzes defect count, material quality, durability, finish quality
    - Real-time classification with 95% accuracy
    - Integrates with existing QC workflows
    
    ### 3. Image Quality Inspection
    Computer vision system for automated visual quality control.
    - Detects surface defects, color inconsistencies, packaging issues
    - Processes 100+ images per hour
    - Supports multiple check types: overall, surface defects, color consistency, packaging
    - 80% accuracy in defect detection
    
    ### 4. Defect Detection System
    Object detection model that localizes and classifies product defects.
    - Detects: scratches, dents, discoloration, cracks, missing parts, misalignment
    - Provides bounding boxes and confidence scores
    - Severity classification: minor, moderate, major
    - Used by 5+ e-commerce companies
    
    ## Our Technology Stack
    - Machine Learning: TensorFlow, PyTorch, Scikit-learn
    - Computer Vision: YOLO, OpenCV
    - Backend: Python, FastAPI
    - Frontend: React, Streamlit
    - Database: PostgreSQL
    - Cloud: AWS
    
    ## Pricing Plans
    
    ### Starter Plan - $15/month
    - Up to 500 products analyzed per month
    - Access to Price Prediction and Quality Classification
    - Email support
    - 99.5% uptime SLA
    
    ### Professional Plan - $35/month
    - Up to 1000 products analyzed per month
    - All 4 ML models included
    - Priority support (24/7)
    - API access
    - Custom integrations
    - 99.9% uptime SLA
    
    ### Enterprise Plan - Custom Pricing
    - Unlimited products
    - Dedicated account manager
    - Custom model training
    - On-premise deployment options
    - White-label solutions
    - 99.99% uptime SLA
    
    ## Customer Success Stories
    
    ### TechGadgets Inc.
    "Reduced product returns by 45% after implementing the quality classification system.
    The ROI was clear within 3 months." - Emmanuel Moses, VP of Operations
    
    ### FashionHub Online
    "The image quality inspection caught defects we would have missed manually.
    Customer satisfaction scores increased by 30%." - Faqua Mich, Quality Manager
    
    ### HomeEssentials Market
    "Price prediction helped us optimize pricing strategy and increase margins by 18%." 
    - Alexis Chen, Pricing Director
    
    ## Company Statistics
    - 100+ active customers
    - 1000+ products analyzed to date
    - 85% customer retention rate
    - Offices in Lagos, Abuja, Asaba
    - 60+ employees worldwide
    - $200K funding (2025)
    
    ## Integration and API
    Our platform offers RESTful APIs for seamless integration:
    - Easy integration with Shopify, WooCommerce, Magento
    - Webhook support for real-time notifications
    - Comprehensive API documentation
    - SDKs available for Python, JavaScript, Java
    
    ## Support and Training
    - 24/7 customer support (Pro and Enterprise)
    - Online knowledge base with 50+ articles
    - Video tutorials and webinars
    - Dedicated onboarding specialist
    - Annual customer conference
    
    ## Compliance and Security
    - GDPR compliant
    - ISO 27001 certified
    - Data encryption at rest and in transit
    - Regular security audits
    - 99.9% data accuracy guarantee
    
    ## Contact Information
    - Website: www.ecommerce-ogeqc.com
    - Email: info@ecommerce-ogeqc.com
    - Sales: sales@ecommerce-ogeqc.com
    - Support: support@ecommerce-ogeqc.com
    - Phone: +2347084684214
    - Address: 123 Innovation Drive, Lagos, Nigeria
    
    ## Working Hours
    - Sales Team: Monday-Friday, 9 AM - 6 PM WAT
    - Support Team: 24/7 (Pro and Enterprise customers)
    - Support Team: Monday-Friday, 9 AM - 6 PM WAT (Starter customers)
    """


    # save to a file
    with open("data/raw/company_knowledge.txt", "w", encoding="utf-8") as f:
        f.write(company_info)

    # Load and split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50,
        separators = ["\n\n", "\n", " ", ""]
    )

    # Split the text into chunks
    chunks = text_splitter.split_text(company_info)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Save the vector store
    vectorstore.save_local("data/vector/company_vectorstore")

    print(" Company knowledge base created successfully!")
    return vectorstore

def load_company_knowledge_base():
    """Load existing vector store"""

    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            "company_vectorstore",
            embeddings,
            allow_dangerous_deserialization = True
        )
        print(" Company knowledge base loaded!")
        return vectorstore
    except:
        print(" Creating new knowledge base...")
        return create_company_knowledge_base()
    
# STEP 2: Define Tools for the Agent
def search_company_info(query: str)-> str:
    """Search the company knowledge base for information"""
    vectorstore = load_company_knowledge_base()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    result = "\n\n".join([doc.page_content for doc in docs])
    return result if result else "No relevant information found."

def get_pricing_info(plan: str)-> str:
    """Get pricing information for different plans"""

    pricing = {
        "starter": "Starter Plan: $15/month - Up to 500 products, Price Prediction & Quality Classification, Email support",
        "professional": "Professional Plan: $35/month - Up to 1000 products, All 4 models, 24/7 support, API access",
        "enterprise": "Enterprise Plan: Custom pricing - Unlimited products, Dedicated manager, Custom training, On-premise options"
    }
    if plan.lower() in pricing:
        return pricing[plan.lower()]
    else:
        return "\n".join(pricing.values())
    
def get_contact_info()-> str:
    """Get company contact information"""
    return """
    Contact Information:
    - Website: www.ecommerce-ogeqc.com
    - Email: info@ecommerce-ogeqc.com
    - Sales: sales@ecommerce-ogeqc.com
    - Support: support@ecommerce-ogeqc.com
    - Phone: +2347084684214
    - Address: 123 Innovation Drive, Lagos, Nigeria
    """

def get_business_hours() -> str:
    """Get company working hours"""
    return """
    Working Hours:
    - Sales Team: Monday-Friday, 9 AM - 6 PM WAT
    - Support (Pro/Enterprise): 24/7
    - Support (Starter): Monday-Friday, 9 AM - 6 PM WAT
    """

# Create tools list
tools = [
    Tool(
        name = "search_company_info",
        func = search_company_info,
        description = "Search the company knowledge base for information about products, services, features, technology, customers, etc. Use this for general questions about the company."
    ),
    Tool(
        name = "get_pricing_info",
        func = get_pricing_info,
        description = "Get pricing information for our plans. Input can be 'starter', 'professional', 'enterprise', or 'all'."
    ),
    Tool(
        name = "get_contact_info",
        func = get_contact_info,
        description = "Get company contact information including email, phone, and address."
    ),
    Tool(
        name = "get_business_hours",
        func = get_business_hours,
        description = "Get company working hours and availability information."
    )
]

# STEP 3: Build the LangGraph Agent
class AgentState(TypedDict):
    """State of the agent"""
    messages: Annotated[Sequence[HumanMessage | AIMessage | SystemMessage], operator.add]
    user_input: str
    chat_history: list

class EcommerceAgent:
    """
    LangGraph-based conversational agent for e-commerce company information
    """
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model = "gpt-4o-mini",
            temperature = 0.7,
            streaming = True
        )
        # Create the agent
        self.agent = create_openai_functions_agent(
            llm = self.llm,
            tools = tools,
            prompt = self._create_prompt()
        )
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent = self.agent,
            tools = tools,
            verbose = True,
            handle_parsing_errors = True,
            max_iterations = 3
        )
        # Message history storage
        self.message_histories = {}

    def _create_prompt(self):
        """Create the agent prompt template"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful customer service assistant for an OGE E-commerce Quality Control Solutions company.
             Your role is to:
        - Provide accurate information about the company's products and services
        - Answer questions about pricing, features, and capabilities
        - Help potential customers understand how our solutions can benefit them
        - Be professional, friendly, and concise
        - Use the available tools to search for specific information when needed

        Company Overview:
        We provide AI-powered quality control solutions including:
        1. Price Prediction (Regression Model)
        2. Quality Classification (Tabular Classification)
        3. Image Quality Inspection (Image Classification)
        4. Defect Detection (Object Detection)

        Always be helpful and if you don't know something, use the search_company_info tool to find the answer.
        If asked about pricing, use the get_pricing_info tool.
        If asked about contact details, use the get_contact_info tool.
        """), 
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name = "agent_scratchpad"),
        ])
    
    def get_session_history(self, session_id: str):
        """Get or create chat history for a session"""
        if session_id not in self.message_histories:
            self.message_histories[session_id] = ChatMessageHistory()
        return self.message_histories[session_id]
    
    def chat(self, user_input: str, session_id: str = "default")-> dict:
        """
        Main chat method
        
        Args:
            user_input: User's message
            session_id: Session identifier for conversation history
            
        Returns:
            dict with response and metadata
        """
        # Get chat history
        history = self.get_session_history(session_id)

        # Prepare input with history
        chat_history_message = history.messages[-10:] # Keep last 10 messages

        # Invoke agent
        result = self.agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history_message
        })

        # Save to history
        history.add_user_message(user_input)
        history.add_ai_message(result["output"])

        return {
            "response": result["output"],
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_history(self, session_id: str = "default"):
        """Clear chat history for a session"""
        if session_id in self.message_histories:
            self.message_histories[session_id].clear()

# STEP 4: Simple CLI Test Interface
def test_chatbot():
    """Test the chatbot in CLI"""
    print("=" * 60)
    print("E-commerce Company Information Chatbot")
    print("=" * 60)
    print("Ask me anything about our company, products, or services!")
    print("Type 'quit' to exit\n")
    
    # Initialize agent
    agent = EcommerceAgent()
    session_id = "test_session"

    while True:
        user_input = input("\n You: ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\n Thank you for chatting! Have a great day!")
            break
        if not user_input:
            continue

        # Get response
        print("\n Assistant: ", end="", flush=True)
        result = agent.chat(user_input, session_id)
        print(result["response"])

    
if __name__ == "__main__":
    # Create knowledge base if it doesn't exist
    if not os.path.exists("data/vector/company_vectorstore"):
        print("Creating company knowledge base...")
        create_company_knowledge_base()
    
    # Run test
    test_chatbot()
