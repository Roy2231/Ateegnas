import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Textarea } from './components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Badge } from './components/ui/badge';
import { Progress } from './components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './components/ui/dialog';
import { Label } from './components/ui/label';
import { Checkbox } from './components/ui/checkbox';
import { AlertCircle, Upload, Download, Play, Terminal, Database, Cpu, CheckCircle, Clock, X, Plus } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const AGENT_TYPES = [
  { id: 'entity_extraction', name: 'Entity Extraction', icon: '🏷️' },
  { id: 'ner', name: 'Named Entity Recognition', icon: '👤' },
  { id: 'sentiment_analysis', name: 'Sentiment Analysis', icon: '😊' },
  { id: 'text_classification', name: 'Text Classification', icon: '📂' },
  { id: 'translation', name: 'Translation', icon: '🌐' },
  { id: 'summarization', name: 'Summarization', icon: '📄' }
];

function App() {
  const [projects, setProjects] = useState([]);
  const [selectedProject, setSelectedProject] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  
  // Project creation state
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newProject, setNewProject] = useState({
    name: '',
    description: '',
    ontology: '',
    selected_agents: []
  });

  useEffect(() => {
    loadProjects();
  }, []);

  useEffect(() => {
    if (selectedProject) {
      loadResults(selectedProject.id);
      const interval = setInterval(() => {
        if (selectedProject.status === 'processing') {
          refreshProject(selectedProject.id);
        }
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [selectedProject]);

  const loadProjects = async () => {
    try {
      const response = await axios.get(`${API}/projects`);
      setProjects(response.data);
    } catch (error) {
      console.error('Error loading projects:', error);
    }
  };

  const refreshProject = async (projectId) => {
    try {
      const response = await axios.get(`${API}/projects/${projectId}`);
      setSelectedProject(response.data);
      await loadProjects();
    } catch (error) {
      console.error('Error refreshing project:', error);
    }
  };

  const loadResults = async (projectId) => {
    try {
      const response = await axios.get(`${API}/projects/${projectId}/results`);
      setResults(response.data);
    } catch (error) {
      console.error('Error loading results:', error);
    }
  };

  const createProject = async () => {
    if (!newProject.name || !newProject.ontology || newProject.selected_agents.length === 0) {
      return;
    }

    try {
      setLoading(true);
      const response = await axios.post(`${API}/projects`, newProject);
      setProjects([response.data, ...projects]);
      setCreateDialogOpen(false);
      setNewProject({ name: '', description: '', ontology: '', selected_agents: [] });
    } catch (error) {
      console.error('Error creating project:', error);
    } finally {
      setLoading(false);
    }
  };

  const uploadFile = async (file) => {
    if (!selectedProject || !file) return;

    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('file', file);
      
      await axios.post(`${API}/projects/${selectedProject.id}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      await refreshProject(selectedProject.id);
    } catch (error) {
      console.error('Error uploading file:', error);
    } finally {
      setLoading(false);
    }
  };

  const exportResults = async () => {
    if (!selectedProject) return;

    try {
      const response = await axios.get(`${API}/projects/${selectedProject.id}/export`);
      const blob = new Blob([JSON.stringify(response.data, null, 2)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedProject.name}_annotations.json`;
      a.click();
    } catch (error) {
      console.error('Error exporting results:', error);
    }
  };

  const deleteProject = async (projectId) => {
    try {
      await axios.delete(`${API}/projects/${projectId}`);
      setProjects(projects.filter(p => p.id !== projectId));
      if (selectedProject?.id === projectId) {
        setSelectedProject(null);
      }
    } catch (error) {
      console.error('Error deleting project:', error);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'processing': return <Clock className="w-4 h-4 text-blue-400 animate-spin" />;
      case 'failed': return <AlertCircle className="w-4 h-4 text-red-400" />;
      default: return <Database className="w-4 h-4 text-gray-400" />;
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <Terminal className="w-8 h-8 text-cyan-400" />
            <h1 className="app-title">Ateegnas</h1>
            <span className="app-subtitle">Multi-Agent AI Data Annotation</span>
          </div>
          
          <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button className="create-btn">
                <Plus className="w-4 h-4 mr-2" />
                New Project
              </Button>
            </DialogTrigger>
            <DialogContent className="dialog-content">
              <DialogHeader>
                <DialogTitle>Create New Annotation Project</DialogTitle>
              </DialogHeader>
              <div className="form-grid">
                <div>
                  <Label htmlFor="project-name">Project Name</Label>
                  <Input
                    id="project-name"
                    value={newProject.name}
                    onChange={(e) => setNewProject({...newProject, name: e.target.value})}
                    placeholder="Enter project name..."
                    className="form-input"
                  />
                </div>
                
                <div>
                  <Label htmlFor="project-desc">Description (Optional)</Label>
                  <Input
                    id="project-desc"
                    value={newProject.description}
                    onChange={(e) => setNewProject({...newProject, description: e.target.value})}
                    placeholder="Project description..."
                    className="form-input"
                  />
                </div>
                
                <div className="col-span-2">
                  <Label htmlFor="ontology">Ontology / Annotation Instructions</Label>
                  <Textarea
                    id="ontology"
                    value={newProject.ontology}
                    onChange={(e) => setNewProject({...newProject, ontology: e.target.value})}
                    placeholder="Define your annotation schema, labels, or instructions..."
                    className="form-textarea"
                    rows="4"
                  />
                </div>
                
                <div className="col-span-2">
                  <Label>Select Annotation Agents</Label>
                  <div className="agents-grid">
                    {AGENT_TYPES.map((agent) => (
                      <div key={agent.id} className="agent-checkbox">
                        <Checkbox
                          id={agent.id}
                          checked={newProject.selected_agents.includes(agent.id)}
                          onCheckedChange={(checked) => {
                            if (checked) {
                              setNewProject({
                                ...newProject,
                                selected_agents: [...newProject.selected_agents, agent.id]
                              });
                            } else {
                              setNewProject({
                                ...newProject,
                                selected_agents: newProject.selected_agents.filter(id => id !== agent.id)
                              });
                            }
                          }}
                        />
                        <label htmlFor={agent.id} className="agent-label">
                          <span className="agent-icon">{agent.icon}</span>
                          {agent.name}
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="col-span-2 flex justify-end gap-3 mt-4">
                  <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button onClick={createProject} disabled={loading}>
                    {loading ? 'Creating...' : 'Create Project'}
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </header>

      <div className="main-container">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <h3>Projects</h3>
            <Badge variant="secondary">{projects.length}</Badge>
          </div>
          
          <div className="projects-list">
            {projects.map((project) => (
              <div
                key={project.id}
                className={`project-item ${selectedProject?.id === project.id ? 'active' : ''}`}
                onClick={() => setSelectedProject(project)}
              >
                <div className="project-header">
                  <div className="project-title">{project.name}</div>
                  <div className="project-actions">
                    {getStatusIcon(project.status)}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteProject(project.id);
                      }}
                      className="delete-btn"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                </div>
                
                <div className="project-meta">
                  <span className="agent-count">
                    <Cpu className="w-3 h-3" />
                    {project.selected_agents.length} agents
                  </span>
                  <span className={`status-badge ${project.status}`}>
                    {project.status}
                  </span>
                </div>
                
                {project.total_records > 0 && (
                  <div className="progress-container">
                    <Progress 
                      value={(project.processed_records / project.total_records) * 100} 
                      className="progress-bar"
                    />
                    <span className="progress-text">
                      {project.processed_records}/{project.total_records}
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </aside>

        {/* Main Content */}
        <main className="main-content">
          {selectedProject ? (
            <Tabs value={activeTab} onValueChange={setActiveTab} className="project-tabs">
              <TabsList className="tabs-list">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="upload">Upload Data</TabsTrigger>
                <TabsTrigger value="results">Results</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="tab-content">
                <Card className="project-overview">
                  <CardHeader>
                    <CardTitle>{selectedProject.name}</CardTitle>
                    {selectedProject.description && (
                      <p className="project-description">{selectedProject.description}</p>
                    )}
                  </CardHeader>
                  <CardContent>
                    <div className="overview-grid">
                      <div className="overview-section">
                        <h4>Ontology</h4>
                        <div className="ontology-box">
                          {selectedProject.ontology}
                        </div>
                      </div>
                      
                      <div className="overview-section">
                        <h4>Active Agents</h4>
                        <div className="agents-list">
                          {selectedProject.selected_agents.map((agentId) => {
                            const agent = AGENT_TYPES.find(a => a.id === agentId);
                            return (
                              <Badge key={agentId} variant="outline" className="agent-badge">
                                {agent?.icon} {agent?.name} <span className="scale-indicator">×50</span>
                              </Badge>
                            );
                          })}
                        </div>
                        <p className="scaling-info">
                          🚀 <strong>50x Scale:</strong> Each agent type runs 50 parallel instances 
                          = {selectedProject.selected_agents.length * 50} total agents
                        </p>
                      </div>
                      
                      <div className="overview-stats">
                        <div className="stat-item">
                          <span className="stat-value">{selectedProject.total_records}</span>
                          <span className="stat-label">Total Records</span>
                        </div>
                        <div className="stat-item">
                          <span className="stat-value">{selectedProject.processed_records}</span>
                          <span className="stat-label">Processed</span>
                        </div>
                        <div className="stat-item">
                          <span className={`stat-value status-${selectedProject.status}`}>
                            {selectedProject.status}
                          </span>
                          <span className="stat-label">Status</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="upload" className="tab-content">
                <Card className="upload-section">
                  <CardHeader>
                    <CardTitle>Upload Dataset</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="upload-area">
                      <Upload className="upload-icon" />
                      <h3>Drag & Drop or Click to Upload</h3>
                      <p>Supports CSV, JSON, PDF, and TXT files</p>
                      <input
                        type="file"
                        accept=".csv,.json,.pdf,.txt"
                        onChange={(e) => uploadFile(e.target.files[0])}
                        className="file-input"
                        disabled={loading}
                      />
                      {loading && <div className="loading-spinner">Processing...</div>}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="results" className="tab-content">
                <Card className="results-section">
                  <CardHeader>
                    <div className="results-header">
                      <CardTitle>Annotation Results</CardTitle>
                      {results.length > 0 && (
                        <Button onClick={exportResults} className="export-btn">
                          <Download className="w-4 h-4 mr-2" />
                          Export Results
                        </Button>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent>
                    {results.length === 0 ? (
                      <div className="empty-state">
                        <Database className="empty-icon" />
                        <h3>No Results Yet</h3>
                        <p>Upload a dataset to start annotation</p>
                      </div>
                    ) : (
                      <div className="results-grid">
                        {results.map((result, index) => (
                          <div key={result.id} className="result-item">
                            <div className="result-header">
                              <span className="result-index">#{index + 1}</span>
                              {result.quality_score && (
                                <Badge variant="outline" className="quality-badge">
                                  Q: {(result.quality_score * 100).toFixed(0)}%
                                </Badge>
                              )}
                            </div>
                            <div className="result-text">
                              {result.original_text.substring(0, 200)}...
                            </div>
                            <div className="result-annotations">
                              {Object.entries(result.agent_results).map(([agentType, agentResult]) => (
                                <div key={agentType} className="annotation-result">
                                  <div className="annotation-header">
                                    {AGENT_TYPES.find(a => a.id === agentType)?.icon}
                                    <span>{agentType.replace('_', ' ')}</span>
                                    <Badge variant="secondary" className="instance-count">
                                      50× Consensus
                                    </Badge>
                                  </div>
                                  <div className="annotation-content">
                                    {agentResult.consensus_result ? (
                                      <div className="consensus-display">
                                        <div className="consensus-header">
                                          <span className="consensus-label">Consensus Result:</span>
                                          <span className="instance-info">
                                            From {agentResult.instance_count || 50} instances
                                          </span>
                                        </div>
                                        <pre className="consensus-result">
                                          {JSON.stringify(agentResult.consensus_result, null, 2)}
                                        </pre>
                                        {agentResult.aggregated_results && (
                                          <details className="raw-results">
                                            <summary>View all {agentResult.aggregated_results.length} individual results</summary>
                                            <pre className="raw-data">
                                              {JSON.stringify(agentResult.aggregated_results, null, 2)}
                                            </pre>
                                          </details>
                                        )}
                                      </div>
                                    ) : (
                                      <pre>{JSON.stringify(agentResult.result || agentResult, null, 2)}</pre>
                                    )}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          ) : (
            <div className="welcome-section">
              <Terminal className="welcome-icon" />
              <h2>Welcome to Ateegnas</h2>
              <p>Multi-Agent AI Data Annotation Platform</p>
              <p className="welcome-description">
                Select a project from the sidebar to get started, or create a new one to begin annotating your data with AI agents.
              </p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;