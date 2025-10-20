const { useState, useEffect } = React;

function AlertSystem() {
    const [alerts, setAlerts] = useState([]);
    const [showNotification, setShowNotification] = useState(false);

    useEffect(() => {
        // Poll for sensor data and predictions
        const pollInterval = setInterval(async () => {
            try {
                const [sensorRes, predictionRes] = await Promise.all([
                    fetch('/api/sensor-data'),
                    fetch('/api/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            dem_data: window.demData,
                            image_data: window.latestImageData,
                            sensor_data: window.latestSensorData
                        })
                    })
                ]);

                const [sensorData, prediction] = await Promise.all([
                    sensorRes.json(),
                    predictionRes.json()
                ]);

                // Check thresholds and predictions
                const newAlerts = checkThresholds(sensorData, prediction);
                if (newAlerts.length > 0) {
                    setAlerts(prev => [...newAlerts, ...prev].slice(0, 10)); // Keep last 10 alerts
                    setShowNotification(true);
                    playAlertSound();
                }
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }, 30000); // Poll every 30 seconds

        return () => clearInterval(pollInterval);
    }, []);

    const checkThresholds = (sensorData, prediction) => {
        const thresholds = {
            displacement: 10.0, // mm
            strain: 0.002,     // strain ratio
            pore_pressure: 50, // kPa
            rainfall: 100,     // mm/hr
            vibration: 5.0     // m/s²
        };

        const newAlerts = [];
        
        // Check sensor thresholds
        for (const [sensor, value] of Object.entries(sensorData)) {
            if (value > thresholds[sensor]) {
                newAlerts.push({
                    id: Date.now() + Math.random(),
                    type: 'warning',
                    message: `${sensor.toUpperCase()} exceeded threshold: ${value}`,
                    timestamp: new Date().toLocaleTimeString()
                });
            }
        }

        // Check rockfall probability
        if (prediction.probability > 0.7) {
            newAlerts.push({
                id: Date.now() + Math.random(),
                type: 'danger',
                message: `High rockfall probability detected: ${(prediction.probability * 100).toFixed(1)}%`,
                timestamp: new Date().toLocaleTimeString()
            });
        }

        return newAlerts;
    };

    const playAlertSound = () => {
        const audio = new Audio('/alert-sound.mp3');
        audio.play().catch(err => console.error('Error playing alert sound:', err));
    };

    return (
        <div className="fixed bottom-4 right-4 z-50">
            {showNotification && alerts.length > 0 && (
                <div className="max-w-sm w-full bg-red-600 text-white rounded-lg shadow-lg p-4 mb-4 animate-bounce">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center">
                            <i className="icon-alert-triangle mr-2"></i>
                            <span className="font-semibold">New Alert</span>
                        </div>
                        <button 
                            onClick={() => setShowNotification(false)}
                            className="text-white hover:text-gray-200"
                        >
                            <i className="icon-x"></i>
                        </button>
                    </div>
                </div>
            )}

            <div className="max-h-96 overflow-y-auto bg-glass-bg backdrop-blur-md rounded-lg shadow-lg">
                <div className="p-4">
                    <h3 className="text-lg font-semibold mb-2 text-white">Recent Alerts</h3>
                    {alerts.length === 0 ? (
                        <p className="text-gray-400">No recent alerts</p>
                    ) : (
                        <div className="space-y-2">
                            {alerts.map(alert => (
                                <div 
                                    key={alert.id}
                                    className={`flex items-center justify-between p-2 rounded ${
                                        alert.type === 'danger' ? 'bg-red-600/20' : 'bg-yellow-600/20'
                                    }`}
                                >
                                    <div>
                                        <p className="text-white">{alert.message}</p>
                                        <p className="text-sm text-gray-400">{alert.timestamp}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default AlertSystem;
}
