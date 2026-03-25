import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { fetchGraphRoot, expandNode } from '../api/client';

const NODE_COLORS = {
  Customer: '#7c3aed',
  SalesOrder: '#2563eb',
  SalesOrderItem: '#0891b2',
  Delivery: '#0d9488',
  DeliveryItem: '#16a34a',
  BillingDocument: '#d97706',
  BillingItem: '#ea580c',
  JournalEntry: '#e11d48',
  Payment: '#db2777',
  Product: '#65a30d',
  Plant: '#64748b',
};

const CORE_TYPES = ['Customer', 'SalesOrder', 'Delivery', 'BillingDocument', 'JournalEntry', 'Payment'];

const LEGEND_ITEMS = Object.entries(NODE_COLORS).map(([type, color]) => ({
  type,
  color,
  label: type.replace(/([A-Z])/g, ' $1').trim(),
}));

export default function GraphView({ onNodeSelect, highlightedNodes, minimized, showGranular, onToggleMinimize, onToggleGranular }) {
  const fgRef = useRef();
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState(null);
  const [containerDimensions, setContainerDimensions] = useState({ width: 800, height: 600 });
  const containerRef = useRef(null);
  const mousePos = useRef({ x: 0, y: 0 });
  const [tooltip, setTooltip] = useState(null);
  const hasZoomed = useRef(false);

  const isFlowMode = highlightedNodes?.length > 0;

  useEffect(() => {
    if (isFlowMode && fgRef.current) {
      const timer = setTimeout(() => {
        if (fgRef.current) fgRef.current.zoomToFit(800, 150);
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [highlightedNodes, isFlowMode]);

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setContainerDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight
        });
      }
    };
    window.addEventListener('resize', updateDimensions);
    updateDimensions();
    setTimeout(updateDimensions, 100);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  useEffect(() => { loadRoot(); }, []);

  const loadRoot = async () => {
    setLoading(true);
    hasZoomed.current = false;
    try {
      const data = await fetchGraphRoot();
      const nodes = (data.nodes || []).map((n) => ({
        ...n,
        id: n.id,
        val: n.node_type === 'Customer' ? 8 : CORE_TYPES.includes(n.node_type) ? 4 : 2,
        color: NODE_COLORS[n.node_type] || '#94a3b8',
      }));

      const links = (data.edges || []).map(e => ({
        source: e.source,
        target: e.target,
        type: e.type,
        id: `${e.source}-${e.target}`
      }));

      setGraphData({ nodes, links });
    } catch (e) {
      console.error('Failed to load root graph:', e);
    }
    setLoading(false);
  };

  const handleNodeClick = useCallback(async (node) => {
    onNodeSelect?.(node.id, node);

    if (fgRef.current) {
      fgRef.current.centerAt(node.x, node.y, 400);
      fgRef.current.zoom(2.5, 400);
    }

    try {
      const data = await expandNode(node.id);
      if (!data.nodes?.length) return;

      setGraphData(prev => {
        const existingNodeIds = new Set(prev.nodes.map(n => n.id));
        const existingLinkIds = new Set(prev.links.map(l => l.id || `${l.source.id}-${l.target.id}`));

        const newNodes = data.nodes
          .filter(n => !existingNodeIds.has(n.id))
          .map(n => ({
            ...n,
            id: n.id,
            val: n.node_type === 'Customer' ? 8 : CORE_TYPES.includes(n.node_type) ? 4 : 2,
            color: NODE_COLORS[n.node_type] || '#94a3b8',
            x: node.x + (Math.random() - 0.5) * 50,
            y: node.y + (Math.random() - 0.5) * 50
          }));

        const newLinks = data.edges
          .filter(e => !existingLinkIds.has(`${e.source}-${e.target}`))
          .map(e => ({
            source: e.source, target: e.target, type: e.type,
            id: `${e.source}-${e.target}`
          }));

        const updatedNodes = prev.nodes.map(n =>
          n.id === node.id ? { ...n, expanded: true } : n
        );

        return {
          nodes: [...updatedNodes, ...newNodes],
          links: [...prev.links, ...newLinks]
        };
      });
    } catch (e) {
      console.error('Expand failed:', e);
    }
  }, [onNodeSelect]);

  const resetGraph = async () => {
    setGraphData({ nodes: [], links: [] });
    setFilter(null);
    setTimeout(() => loadRoot(), 50);
  };

  const fitGraph = () => {
    if (fgRef.current) fgRef.current.zoomToFit(400, 50);
  };

  const visibleData = useMemo(() => {
    let data = graphData;

    if (!showGranular) {
      const coreNodeIds = new Set(data.nodes.filter(n => CORE_TYPES.includes(n.node_type)).map(n => n.id));
      data = {
        nodes: data.nodes.filter(n => coreNodeIds.has(n.id)),
        links: data.links.filter(l => {
          const sid = typeof l.source === 'object' ? l.source.id : l.source;
          const tid = typeof l.target === 'object' ? l.target.id : l.target;
          return coreNodeIds.has(sid) && coreNodeIds.has(tid);
        })
      };
    }

    if (!filter) return data;

    const primaryNodes = new Set(data.nodes.filter(n => n.node_type === filter).map(n => n.id));
    const connectedNodeIds = new Set([...primaryNodes]);
    data.links.forEach(l => {
      const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
      const targetId = typeof l.target === 'object' ? l.target.id : l.target;
      if (primaryNodes.has(sourceId)) connectedNodeIds.add(targetId);
      if (primaryNodes.has(targetId)) connectedNodeIds.add(sourceId);
    });

    return {
      nodes: data.nodes.filter(n => connectedNodeIds.has(n.id)),
      links: data.links.filter(l =>
        connectedNodeIds.has(typeof l.source === 'object' ? l.source.id : l.source) &&
        connectedNodeIds.has(typeof l.target === 'object' ? l.target.id : l.target)
      ),
    };
  }, [graphData, filter, showGranular]);

  const handleMouseMove = useCallback((e) => {
    mousePos.current = { x: e.clientX, y: e.clientY };
  }, []);

  return (
    <div
      className="graph-panel"
      ref={containerRef}
      onMouseMove={handleMouseMove}
      style={{ background: '#f8fafc', position: 'relative', width: '100%', height: '100%' }}
    >
      {loading && (
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: '#94a3b8', zIndex: 4, fontWeight: 500 }}>
          Loading graph...
        </div>
      )}

      {containerDimensions.width > 0 && (
        <ForceGraph2D
          ref={fgRef}
          width={containerDimensions.width}
          height={containerDimensions.height}
          backgroundColor="#f8fafc"
          graphData={visibleData}
          dagMode={isFlowMode ? 'lr' : null}
          dagLevelDistance={120}
          nodeLabel=""
          onNodeHover={node => setTooltip(node ? { node, x: mousePos.current.x, y: mousePos.current.y } : null)}
          nodeColor={n => highlightedNodes?.includes(n.id) ? '#2563eb' : n.color}
          nodeRelSize={3}
          linkColor={() => 'rgba(147, 197, 253, 0.5)'}
          linkWidth={l => highlightedNodes?.includes(l.source?.id ?? l.source) ? 2 : 0.5}
          linkDirectionalArrowLength={isFlowMode ? 3.5 : 0}
          linkDirectionalArrowRelPos={1}
          linkDirectionalParticles={isFlowMode ? 2 : 0}
          linkDirectionalParticleWidth={1.5}
          onNodeClick={handleNodeClick}
          cooldownTicks={100}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          onEngineStop={() => {
            if (!hasZoomed.current && fgRef.current) {
              fgRef.current.zoomToFit(400, 50);
              hasZoomed.current = true;
            }
          }}
          nodeCanvasObject={(node, ctx, globalScale) => {
            const r = Math.sqrt(node.val || 4) * (isFlowMode ? 1.8 : 1.2);
            ctx.beginPath();
            ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
            const isHighlighted = highlightedNodes?.includes(node.id);
            ctx.fillStyle = isHighlighted ? '#2563eb' : node.color;
            ctx.globalAlpha = isHighlighted ? 1.0 : 0.8;
            ctx.fill();
            ctx.globalAlpha = 1.0;

            if (isHighlighted) {
              ctx.strokeStyle = '#1d4ed8';
              ctx.lineWidth = 2.0 / globalScale;
              ctx.stroke();
            }

            if ((isFlowMode && globalScale > 1.2) || (isHighlighted && globalScale > 0.8)) {
              const label = node.label || node.id;
              const fontSize = 10 / globalScale;
              ctx.font = `${fontSize}px Inter, sans-serif`;
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              const textWidth = ctx.measureText(label).width;
              ctx.fillStyle = 'rgba(248, 250, 252, 0.9)';
              ctx.fillRect(node.x - textWidth / 2 - 2, node.y + r + 2, textWidth + 4, fontSize + 2);
              ctx.fillStyle = '#1e293b';
              ctx.fillText(label, node.x, node.y + r + fontSize / 2 + 3);
            }
          }}
        />
      )}

      {tooltip && tooltip.node && (
        <div className="graph-tooltip" style={{ top: Math.max(10, tooltip.y - 100), left: tooltip.x + 20 }}>
          <h4>{tooltip.node.node_type.replace(/([A-Z])/g, ' $1').trim()}</h4>
          <div className="graph-tooltip-row">
            <span className="graph-tooltip-label">ID:</span>
            <span>{tooltip.node.raw_id || tooltip.node.id}</span>
          </div>
          {tooltip.node.label && tooltip.node.label !== tooltip.node.id && (
            <div className="graph-tooltip-row">
              <span className="graph-tooltip-label">Label:</span>
              <span>{tooltip.node.label}</span>
            </div>
          )}
          {Object.entries(tooltip.node)
            .filter(([k, v]) => !['id', 'raw_id', 'node_type', 'x', 'y', 'vx', 'vy', 'fx', 'fy', 'index', 'color', 'val', 'level', 'expanded', 'depth', 'label', '__indexColor'].includes(k) && v != null && typeof v !== 'object')
            .slice(0, 6)
            .map(([k, v]) => (
              <div key={k} className="graph-tooltip-row">
                <span className="graph-tooltip-label">{k.replace(/_/g, ' ')}:</span>
                <span style={{ maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', textAlign: 'right' }}>{String(v)}</span>
              </div>
            ))}
        </div>
      )}

      <div className="graph-overlay-buttons">
        <button className="overlay-btn" onClick={onToggleMinimize}>
          {minimized ? '⊞ Expand' : '⊟ Minimize'}
        </button>
        <button className="overlay-btn" onClick={onToggleGranular}>
          {showGranular ? '◉ Hide Granular Overlay' : '○ Show Granular Overlay'}
        </button>
      </div>

      <div className="graph-controls">
        <button onClick={resetGraph}>⟲ Reset</button>
        <button onClick={fitGraph}>⊞ Fit</button>
        {CORE_TYPES.map((type) => (
          <button
            key={type}
            className={filter === type ? 'active' : ''}
            onClick={() => setFilter(filter === type ? null : type)}
            style={{ fontSize: '11px' }}
          >
            {type.replace(/([A-Z])/g, ' $1').trim()}
          </button>
        ))}
      </div>

      <div className="graph-legend" style={{ position: 'absolute', bottom: 12, left: 12 }}>
        {LEGEND_ITEMS.map(({ type, color, label }) => (
          <div key={type} className="legend-item" style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '11px', color: '#475569', fontWeight: 500 }}>
            <div className="legend-dot" style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: color }} />
            {label}
          </div>
        ))}
      </div>
    </div>
  );
}
