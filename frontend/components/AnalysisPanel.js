const { useState, useEffect } = React;

function AnalysisPanel() {
    const [metrics, setMetrics] = useState(null);
    const [sensorData, setSensorData] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [historicalData, setHistoricalData] = useState(null);

    useEffect(() => {
        // Handle real-time analysis updates
        function handleAnalysisUpdate(event) {
            setMetrics(event.detail);
        }

        window.addEventListener('analysis-update', handleAnalysisUpdate);

        // Poll for sensor data and predictions
        const pollInterval = setInterval(async () => {
            try {
                const [sensorRes, historicalRes] = await Promise.all([
                    fetch('/api/sensor-data'),
                    fetch('/api/historical-data?hours=24')
                ]);

                const [currentSensorData, historicalSensorData] = await Promise.all([
                    sensorRes.json(),
                    historicalRes.json()
                ]);

                setSensorData(currentSensorData);
                setHistoricalData(historicalSensorData);

                // Get prediction if we have all required data
                if (window.demData && window.latestImageData) {
                    const predictionRes = await fetch('/api/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            dem_data: window.demData,
                            image_data: window.latestImageData,
                            sensor_data: currentSensorData
                        })
                    });
                    const predictionData = await predictionRes.json();
                    setPrediction(predictionData);
                }
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }, 30000);

        return () => {
            window.removeEventListener('analysis-update', handleAnalysisUpdate);
            clearInterval(pollInterval);
        };
    }, []);

    const getRiskLevel = (probability) => {
        if (probability > 0.7) return { level: 'High', color: 'text-red-500' };
        if (probability > 0.4) return { level: 'Medium', color: 'text-yellow-500' };
        return { level: 'Low', color: 'text-green-500' };
    };

    return (
        <div className="glass-panel p-4 flex-grow animate-slide-up overflow-y-auto">
            <h2 className="text-xl font-bold mb-4">Risk Analysis</h2>
            
            {/* Current Risk Assessment */}
            {prediction && (
                <div className="mb-6 bg-glass-bg p-4 rounded-lg">
                    <h3 className="font-semibold mb-2">Current Risk Assessment</h3>
                    <div className="flex items-center justify-between">
                        <span>Risk Level:</span>
                        <span className={getRiskLevel(prediction.probability).color + " font-bold"}>
                            {getRiskLevel(prediction.probability).level}
                            ({(prediction.probability * 100).toFixed(1)}%)
                        </span>
                    </div>
                </div>
            )}
            
            {/* Sensor Readings */}
            {sensorData && (
                <div className="mb-6">
                    <h3 className="font-semibold mb-2">Current Sensor Readings</h3>
                    <div className="grid grid-cols-2 gap-4">
                        {Object.entries(sensorData).map(([sensor, value]) => (
                            <div key={sensor} className="bg-glass-bg p-3 rounded-lg">
                                <p className="text-sm text-text-muted">{sensor.replace('_', ' ').toUpperCase()}</p>
                                <p className="text-lg font-bold">{typeof value === 'number' ? value.toFixed(2) : value}</p>
                            </div>
                        ))}
                    </div>
                </div>
            )}
            
            {/* Environmental Metrics */}
            {metrics && (
                <div className="mb-6">
                    <h3 className="font-semibold mb-2">Area Analysis</h3>
                    <div className="flex justify-around mb-4">
                        <div>
                            <p className="text-sm text-text-muted">Max Risk</p>
                            <p className="text-2xl font-bold text-primary-color">{metrics.max_risk.toFixed(2)}</p>
                        </div>
                        <div>
                            <p className="text-sm text-text-muted">Mean Risk</p>
                            <p className="text-2xl font-bold text-secondary-color">{metrics.mean_risk.toFixed(2)}</p>
                        </div>
                    </div>
                    <h4 className="font-semibold mt-4 mb-2">Contributing Factors:</h4>
                    <ul className="space-y-2">
                        {Object.entries(metrics.factors).map(([key, value]) => (
                            <li key={key} className="flex justify-between items-center bg-gray-800/50 p-2 rounded-lg">
                                <span className="text-text-secondary">{key}</span>
                                <span className="font-semibold">{typeof value === 'number' ? value.toFixed(2) : value}</span>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
            
            {!metrics && !sensorData && !prediction && (
                <p className="text-text-muted">Loading analysis data...</p>
            )}
        </div>
    );
}

export default AnalysisPanel;
