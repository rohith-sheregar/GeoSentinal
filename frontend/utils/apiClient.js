export const backend = "http://127.0.0.1:5000";

export async function fetchBounds() {
    const response = await fetch(`${backend}/bounds`);
    if (!response.ok) {
        throw new Error('Failed to fetch map bounds');
    }
    return await response.json();
}

export async function fetchPolygonHeatmap(coordinates) {
    const response = await fetch(`${backend}/polygon_heatmap`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ coordinates })
    });
    if (!response.ok) {
        throw new Error('Network response was not ok');
    }
    const imageBlob = await response.blob();
    return URL.createObjectURL(imageBlob);
}

export async function fetchPolygonMetrics(coordinates) {
    const response = await fetch(`${backend}/polygon_metrics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ coordinates })
    });
    if (!response.ok) {
        throw new Error('Failed to fetch polygon metrics');
    }
    return await response.json();
}

export async function fetchRiskGraph(factors = {}) {
    const response = await fetch(`${backend}/risk_graph`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ factors })
    });
    if (!response.ok) {
        throw new Error('Failed to load graph data');
    }
    return await response.json();
}