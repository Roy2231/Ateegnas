from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import json
import csv
import io
import asyncio
from enum import Enum
import PyPDF2
from openai import AsyncOpenAI

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# OpenAI client
openai_client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

app = FastAPI(title="Ateegnas - Multi-Agent AI Data Annotation")
api_router = APIRouter(prefix="/api")

# Models
class AgentType(str, Enum):
    ENTITY_EXTRACTION = "entity_extraction"
    NER = "ner"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TEXT_CLASSIFICATION = "text_classification"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"

class ProjectStatus(str, Enum):
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Project(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    ontology: str
    selected_agents: List[AgentType]
    status: ProjectStatus = ProjectStatus.CREATED
    total_records: int = 0
    processed_records: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    ontology: str
    selected_agents: List[AgentType]

class AnnotationResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    record_index: int
    original_text: str
    agent_results: Dict[str, Any]
    quality_score: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Agent System
class AnnotationAgent:
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.instance_id = 0  # Will be set during initialization
        
    async def annotate(self, text: str, ontology: str) -> Dict[str, Any]:
        prompts = {
            AgentType.ENTITY_EXTRACTION: f"""Extract entities from the following text based on this ontology: {ontology}
            
Text: {text}

Instance ID: {self.instance_id}/50 - Provide unique perspective in your analysis.
Return the result as a JSON object with extracted entities categorized according to the ontology.""",
            
            AgentType.NER: f"""Perform Named Entity Recognition on the following text. Identify and classify named entities (PERSON, ORGANIZATION, LOCATION, MISC, etc.).
            
Text: {text}

Instance ID: {self.instance_id}/50 - Focus on different aspects and nuances.
Return the result as a JSON object with entities and their classifications.""",
            
            AgentType.SENTIMENT_ANALYSIS: f"""Analyze the sentiment of the following text. Provide sentiment classification and confidence score.
            
Text: {text}

Instance ID: {self.instance_id}/50 - Examine different emotional aspects and subtleties.
Return the result as a JSON object with sentiment (positive/negative/neutral) and confidence score.""",
            
            AgentType.TEXT_CLASSIFICATION: f"""Classify the following text according to this ontology: {ontology}
            
Text: {text}

Instance ID: {self.instance_id}/50 - Consider different classification angles.
Return the result as a JSON object with classification category and confidence score.""",
            
            AgentType.TRANSLATION: f"""Translate the following text to English if it's in another language. If already in English, detect the language.
            
Text: {text}

Instance ID: {self.instance_id}/50 - Pay attention to cultural nuances and context.
Return the result as a JSON object with detected_language, translated_text (if needed), and confidence.""",
            
            AgentType.SUMMARIZATION: f"""Create a concise summary of the following text, focusing on key points mentioned in the ontology: {ontology}
            
Text: {text}

Instance ID: {self.instance_id}/50 - Emphasize different aspects and angles in your summary.
Return the result as a JSON object with summary and key_points."""
        }
        
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are specialized AI annotation agent #{self.instance_id} of 50 instances. Provide unique analytical perspective while maintaining consistency with your agent type's core function. Always return valid JSON responses."
                    },
                    {
                        "role": "user",
                        "content": prompts[self.agent_type]
                    }
                ],
                temperature=0.3 + (self.instance_id * 0.01),  # Slight temperature variation for diversity
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # If not JSON, wrap the response
                result = {
                    "result": response_text, 
                    "agent": self.agent_type.value,
                    "instance_id": self.instance_id
                }
            
            return {
                "agent_type": self.agent_type.value,
                "instance_id": self.instance_id,
                "result": result,
                "status": "success"
            }
        except Exception as e:
            return {
                "agent_type": self.agent_type.value,
                "instance_id": self.instance_id,
                "result": {"error": str(e)},
                "status": "error"
            }

class QualityAssuranceAgent:
    def __init__(self):
        pass
    
    async def evaluate_quality(self, original_text: str, annotations: Dict[str, Any]) -> float:
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quality assurance agent. Evaluate annotation quality on a scale of 0.0 to 1.0 based on accuracy, completeness, and consistency."
                    },
                    {
                        "role": "user",
                        "content": f"""Evaluate the quality of these annotations for the original text. Return only a single number between 0.0 and 1.0 representing quality score.

Original text: {original_text}

Annotations: {json.dumps(annotations, indent=2)}

Quality score (0.0-1.0):"""
                    }
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract score
            try:
                score = float(response_text)
                return max(0.0, min(1.0, score))
            except:
                return 0.5  # Default middle score if parsing fails
        except Exception as e:
            logging.error(f"QA evaluation error: {e}")
            return 0.5

# Processing Functions
async def parse_uploaded_file(file_content: bytes, filename: str) -> List[str]:
    """Parse uploaded file and extract text data"""
    texts = []
    
    try:
        if filename.endswith('.csv'):
            content = file_content.decode('utf-8')
            csv_reader = csv.reader(io.StringIO(content))
            for row in csv_reader:
                texts.extend([cell for cell in row if cell.strip()])
                
        elif filename.endswith('.json'):
            content = file_content.decode('utf-8')
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        # Extract text from common fields
                        for key in ['text', 'content', 'description', 'message']:
                            if key in item:
                                texts.append(str(item[key]))
                                break
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        texts.append(value)
                        
        elif filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():
                    # Split into sentences for better processing
                    sentences = text.split('.')
                    texts.extend([s.strip() for s in sentences if s.strip()])
                    
        elif filename.endswith('.txt'):
            content = file_content.decode('utf-8')
            # Split into paragraphs or sentences
            lines = content.split('\n')
            texts.extend([line.strip() for line in lines if line.strip()])
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")
    
    return [text for text in texts if len(text) > 10]  # Filter short texts

async def process_project_background(project_id: str):
    """Background task to process project annotations with 50x agent scaling"""
    try:
        # Get project
        project_data = await db.projects.find_one({"id": project_id})
        if not project_data:
            return
        
        project = Project(**project_data)
        
        # Update status to processing
        await db.projects.update_one(
            {"id": project_id},
            {"$set": {"status": ProjectStatus.PROCESSING}}
        )
        
        # Get uploaded data
        data_records = await db.project_data.find({"project_id": project_id}).to_list(None)
        
        # Initialize 50x agents for massive parallel processing (300 total agents)
        AGENT_INSTANCES = 50  # 50 instances per agent type
        all_agents = []
        
        for agent_type in project.selected_agents:
            for instance_id in range(AGENT_INSTANCES):
                agent = AnnotationAgent(agent_type)
                agent.instance_id = instance_id  # Add instance tracking
                all_agents.append(agent)
        
        qa_agent = QualityAssuranceAgent()
        
        logging.info(f"Initialized {len(all_agents)} agents ({AGENT_INSTANCES} instances per type)")
        
        # Process each record with distributed agent load
        for i, record in enumerate(data_records):
            text = record['text']
            
            # Distribute workload across all agent instances
            agent_tasks = []
            agent_results = {}
            
            # Group agents by type for result aggregation
            agents_by_type = {}
            for agent in all_agents:
                if agent.agent_type not in agents_by_type:
                    agents_by_type[agent.agent_type] = []
                agents_by_type[agent.agent_type].append(agent)
            
            # Create annotation tasks for each agent type
            for agent_type, type_agents in agents_by_type.items():
                # Distribute text processing across 50 instances of this agent type
                type_tasks = [agent.annotate(text, project.ontology) for agent in type_agents]
                agent_tasks.extend(type_tasks)
            
            # Run all 300 agents in parallel
            logging.info(f"Processing record {i+1} with {len(agent_tasks)} parallel agent instances")
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Aggregate results by agent type (combine results from 50 instances)
            for result in results:
                if not isinstance(result, Exception) and result.get('status') == 'success':
                    agent_type = result['agent_type']
                    
                    if agent_type not in agent_results:
                        agent_results[agent_type] = {
                            'agent_type': agent_type,
                            'aggregated_results': [],
                            'instance_count': 0,
                            'consensus_result': None,
                            'status': 'success'
                        }
                    
                    agent_results[agent_type]['aggregated_results'].append(result['result'])
                    agent_results[agent_type]['instance_count'] += 1
            
            # Create consensus results from 50 instances per agent type
            for agent_type, aggregated_data in agent_results.items():
                instance_results = aggregated_data['aggregated_results']
                
                # Simple consensus: combine all results and create summary
                consensus_result = {
                    'consensus_from_instances': len(instance_results),
                    'combined_analysis': instance_results,
                    'agent_type': agent_type,
                    'processing_scale': f"{AGENT_INSTANCES}x parallel instances"
                }
                
                # For specific agent types, create enhanced consensus
                if agent_type == 'sentiment_analysis':
                    sentiments = []
                    confidences = []
                    for res in instance_results:
                        if isinstance(res, dict) and 'sentiment' in str(res).lower():
                            try:
                                if 'positive' in str(res).lower():
                                    sentiments.append('positive')
                                elif 'negative' in str(res).lower():
                                    sentiments.append('negative')
                                else:
                                    sentiments.append('neutral')
                            except:
                                pass
                    
                    # Consensus sentiment
                    if sentiments:
                        consensus_sentiment = max(set(sentiments), key=sentiments.count)
                        consensus_result['consensus_sentiment'] = consensus_sentiment
                        consensus_result['agreement_ratio'] = sentiments.count(consensus_sentiment) / len(sentiments)
                
                elif agent_type == 'entity_extraction':
                    all_entities = []
                    for res in instance_results:
                        if isinstance(res, dict):
                            try:
                                entities = res.get('entities', []) if 'entities' in res else []
                                all_entities.extend(entities)
                            except:
                                pass
                    
                    # Count entity frequency for consensus
                    entity_counts = {}
                    for entity in all_entities:
                        entity_str = str(entity)
                        entity_counts[entity_str] = entity_counts.get(entity_str, 0) + 1
                    
                    consensus_result['consensus_entities'] = entity_counts
                    consensus_result['total_entity_extractions'] = len(all_entities)
                
                agent_results[agent_type]['consensus_result'] = consensus_result
            
            # Quality assessment on consensus results
            consensus_only = {k: v['consensus_result'] for k, v in agent_results.items()}
            quality_score = await qa_agent.evaluate_quality(text, consensus_only)
            
            # Store annotation result with 50x scaling metadata
            annotation = AnnotationResult(
                project_id=project_id,
                record_index=i,
                original_text=text,
                agent_results=agent_results,  # Contains both individual and consensus results
                quality_score=quality_score
            )
            
            await db.annotations.insert_one(annotation.dict())
            
            # Update progress
            await db.projects.update_one(
                {"id": project_id},
                {"$set": {"processed_records": i + 1}}
            )
            
            logging.info(f"Completed processing record {i+1} with {len(all_agents)} agents")
        
        # Mark as completed
        await db.projects.update_one(
            {"id": project_id},
            {
                "$set": {
                    "status": ProjectStatus.COMPLETED,
                    "completed_at": datetime.now(timezone.utc),
                    "total_agents_used": len(all_agents),
                    "processing_scale": f"{AGENT_INSTANCES}x instances per agent type"
                }
            }
        )
        
        logging.info(f"Project {project_id} completed with {len(all_agents)} total agents")
        
    except Exception as e:
        logging.error(f"Background processing error: {e}")
        await db.projects.update_one(
            {"id": project_id},
            {"$set": {"status": ProjectStatus.FAILED}}
        )

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Ateegnas - Multi-Agent AI Data Annotation API"}

@api_router.post("/projects", response_model=Project)
async def create_project(project_data: ProjectCreate):
    project = Project(**project_data.dict())
    await db.projects.insert_one(project.dict())
    return project

@api_router.get("/projects", response_model=List[Project])
async def get_projects():
    projects = await db.projects.find().to_list(None)
    return [Project(**project) for project in projects]

@api_router.get("/projects/{project_id}", response_model=Project)
async def get_project(project_id: str):
    project_data = await db.projects.find_one({"id": project_id})
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")
    return Project(**project_data)

@api_router.post("/projects/{project_id}/upload")
async def upload_dataset(
    project_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    project_data = await db.projects.find_one({"id": project_id})
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Parse file
    file_content = await file.read()
    texts = await parse_uploaded_file(file_content, file.filename)
    
    # Store data
    for text in texts:
        await db.project_data.insert_one({
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "text": text,
            "uploaded_at": datetime.now(timezone.utc)
        })
    
    # Update project record count
    await db.projects.update_one(
        {"id": project_id},
        {"$set": {"total_records": len(texts)}}
    )
    
    # Start background processing
    background_tasks.add_task(process_project_background, project_id)
    
    return {
        "message": "Dataset uploaded successfully",
        "records_count": len(texts),
        "status": "processing_started"
    }

@api_router.get("/projects/{project_id}/results")
async def get_project_results(project_id: str):
    annotations = await db.annotations.find({"project_id": project_id}).to_list(None)
    return [AnnotationResult(**annotation) for annotation in annotations]

@api_router.get("/projects/{project_id}/export")
async def export_project(project_id: str):
    project_data = await db.projects.find_one({"id": project_id})
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = Project(**project_data)
    annotations = await db.annotations.find({"project_id": project_id}).to_list(None)
    
    export_data = {
        "project": project.dict(),
        "annotations": [AnnotationResult(**annotation).dict() for annotation in annotations],
        "export_timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    return export_data

@api_router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    # Delete project and all related data
    await db.projects.delete_one({"id": project_id})
    await db.project_data.delete_many({"project_id": project_id})
    await db.annotations.delete_many({"project_id": project_id})
    return {"message": "Project deleted successfully"}

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()