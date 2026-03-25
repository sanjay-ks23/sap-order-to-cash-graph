import { useState, useEffect, useCallback } from 'react';
import GraphView from './components/GraphView';
import ChatPanel from './components/ChatPanel';
import NodeDetail from './components/NodeDetail';
import { fetchStats } from './api/client';

export default function App() {
  const [selectedNode, setSelectedNode] = useState(null);
  const [highlightedNodes, setHighlightedNodes] = useState([]);
  const [stats, setStats] = useState(null);
  const [graphMinimized, setGraphMinimized] = useState(false);
  const [showGranular, setShowGranular] = useState(true);

  useEffect(() => {
    fetchStats().then(setStats).catch(console.error);
  }, []);

  const handleNodeSelect = useCallback((nodeId, nodeData) => {
    setSelectedNode({ id: nodeId, data: nodeData });
  }, []);

  const handleHighlightNodes = useCallback((nodes) => {
    setHighlightedNodes(nodes);
  }, []);

  return (
    <>
      <header className="header">
        <div className="header-left">
          <div className="header-logo">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <rect width="16" height="16" rx="3" fill="white" fillOpacity="0.2"/>
              <circle cx="5" cy="5" r="2" fill="white"/>
              <circle cx="11" cy="5" r="2" fill="white"/>
              <circle cx="8" cy="11" r="2" fill="white"/>
              <line x1="5" y1="5" x2="11" y2="5" stroke="white" strokeWidth="0.8"/>
              <line x1="11" y1="5" x2="8" y2="11" stroke="white" strokeWidth="0.8"/>
              <line x1="5" y1="5" x2="8" y2="11" stroke="white" strokeWidth="0.8"/>
            </svg>
          </div>
          <h1>
            <span className="breadcrumb-muted">Mapping</span>
            <span className="breadcrumb-sep">/</span>
            Order to Cash
          </h1>
        </div>
        {stats && (
          <div className="header-stats">
            <div className="header-stat">
              <div className="dot" />
              {stats.total_nodes?.toLocaleString()} nodes
            </div>
            <div className="header-stat">
              {stats.total_edges?.toLocaleString()} edges
            </div>
            <div className="header-stat">
              {Object.keys(stats.by_type || {}).length} types
            </div>
          </div>
        )}
      </header>

      <div className="main-layout">
        <div style={{ flex: 1, position: 'relative' }}>
          <GraphView
            onNodeSelect={handleNodeSelect}
            highlightedNodes={highlightedNodes}
            minimized={graphMinimized}
            showGranular={showGranular}
            onToggleMinimize={() => setGraphMinimized(!graphMinimized)}
            onToggleGranular={() => setShowGranular(!showGranular)}
          />
          {selectedNode && (
            <NodeDetail
              nodeId={selectedNode.id}
              nodeData={selectedNode.data}
              onClose={() => setSelectedNode(null)}
            />
          )}
        </div>
        <ChatPanel onHighlightNodes={handleHighlightNodes} />
      </div>
    </>
  );
}
