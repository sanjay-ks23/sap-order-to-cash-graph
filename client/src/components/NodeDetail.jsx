const NODE_COLORS = {
  Customer: '#8b5cf6',
  SalesOrder: '#3b82f6',
  SalesOrderItem: '#06b6d4',
  Delivery: '#14b8a6',
  DeliveryItem: '#22c55e',
  BillingDocument: '#f59e0b',
  BillingItem: '#f97316',
  JournalEntry: '#f43f5e',
  Payment: '#ec4899',
  Product: '#84cc16',
  Plant: '#64748b',
};

const HIDDEN_KEYS = new Set([
  'id', 'label', 'color', 'borderColor', 'size',
  'nodeType', 'node_type',
]);

export default function NodeDetail({ nodeId, nodeData, onClose }) {
  if (!nodeData) return null;

  const nodeType = nodeData.node_type || nodeData.nodeType || 'Unknown';
  const color = NODE_COLORS[nodeType] || '#64748b';

  const properties = Object.entries(nodeData)
    .filter(([key]) => !HIDDEN_KEYS.has(key))
    .filter(([, val]) => val !== '' && val !== null && val !== undefined);

  return (
    <div className="node-detail-overlay">
      <div className="node-detail-header">
        <h3>
          <span
            className="node-type-badge"
            style={{ background: `${color}22`, color }}
          >
            {nodeType}
          </span>
          {nodeData.label || nodeId}
        </h3>
        <button className="close-btn" onClick={onClose}>×</button>
      </div>
      <div className="node-detail-body">
        {properties.map(([key, value]) => (
          <div key={key} className="detail-row">
            <span className="detail-key">{formatKey(key)}</span>
            <span className="detail-value">{formatValue(value)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatKey(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatValue(val) {
  if (typeof val === 'boolean') return val ? 'Yes' : 'No';
  if (typeof val === 'number') return val.toLocaleString();
  if (typeof val === 'string' && val.includes('T00:00:00')) return val.split('T')[0];
  return String(val);
}
