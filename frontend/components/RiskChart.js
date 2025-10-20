const { useEffect, useRef, useState } = React;

function RiskChart() {
    const chartRef = useRef(null);
    const [chart, setChart] = useState(null);
    const [timeRange, setTimeRange] = useState('24h');

    useEffect(() => {
        loadHistoricalData();
        const interval = setInterval(loadHistoricalData, 30000); // Update every 30 seconds
        
        return () => clearInterval(interval);
    }, [timeRange]);

    async function loadHistoricalData() {
        try {
            const hours = timeRange === '24h' ? 24 : timeRange === '7d' ? 168 : 720;
            const [historicalRes, predictionsRes] = await Promise.all([
                fetch(`/api/historical-data?hours=${hours}`),
                fetch('/api/predictions')
            ]);

            const [historicalData, predictionsData] = await Promise.all([
                historicalRes.json(),
                predictionsRes.json()
            ]);

            updateChart(historicalData, predictionsData);
        } catch (error) {
            console.error('Error loading historical data:', error);
        }
    }

    function updateChart(historicalData, predictionsData) {
        if (chart) {
            chart.destroy();
        }

        const timestamps = historicalData.timestamps || Array.from({ length: 24 }, (_, i) => {
            const date = new Date();
            date.setHours(date.getHours() - (24 - i));
            return date.toLocaleTimeString();
        });

        const datasets = [
            {
                label: 'Risk Probability',
                data: predictionsData.probabilities || [],
                borderColor: '#06b6d4',
                backgroundColor: 'rgba(6, 182, 212, 0.2)',
                fill: true,
                tension: 0.4,
                yAxisID: 'y'
            },
            {
                label: 'Rainfall',
                data: historicalData.rainfall || [],
                borderColor: '#60a5fa',
                backgroundColor: 'transparent',
                borderDash: [5, 5],
                tension: 0.4,
                yAxisID: 'y1'
            },
            {
                label: 'Ground Movement',
                data: historicalData.displacement || [],
                borderColor: '#f97316',
                backgroundColor: 'transparent',
                tension: 0.4,
                yAxisID: 'y2'
            }
        ];

        const newChart = new Chart(chartRef.current, {
            type: 'line',
            data: { labels: timestamps, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#cbd5e1',
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: timeRange === '24h' ? 'hour' : timeRange === '7d' ? 'day' : 'week'
                        },
                        ticks: {
                            color: '#cbd5e1'
                        },
                        grid: {
                            color: 'rgba(203, 213, 225, 0.1)'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Risk Probability',
                            color: '#cbd5e1'
                        },
                        ticks: {
                            color: '#cbd5e1'
                        },
                        grid: {
                            color: 'rgba(203, 213, 225, 0.1)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Rainfall (mm)',
                            color: '#cbd5e1'
                        },
                        ticks: {
                            color: '#cbd5e1'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    },
                    y2: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Movement (mm)',
                            color: '#cbd5e1'
                        },
                        ticks: {
                            color: '#cbd5e1'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });

        setChart(newChart);
    }

    return (
        <div className="glass-panel p-4">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold">Historical Analysis</h2>
                <div className="flex space-x-2">
                    {['24h', '7d', '30d'].map(range => (
                        <button
                            key={range}
                            onClick={() => setTimeRange(range)}
                            className={`px-3 py-1 rounded ${
                                timeRange === range
                                    ? 'bg-primary-color text-white'
                                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                            }`}
                        >
                            {range}
                        </button>
                    ))}
                </div>
            </div>
            <div className="h-64">
                <canvas ref={chartRef}></canvas>
            </div>
        </div>
    );
}

export default RiskChart;
                        },
                        grid: {
                            color: 'rgba(203, 213, 225, 0.1)'
                        }
                    }
                }
            }
        });
        setChart(newChart);
    }

    useEffect(() => {
        loadRiskGraph();

        function handleAnalysisUpdate(event) {
            loadRiskGraph(event.detail.factors);
        }

        window.addEventListener('analysis-update', handleAnalysisUpdate);

        return () => {
            window.removeEventListener('analysis-update', handleAnalysisUpdate);
            if(chart) chart.destroy();
        };
    }, []);

    return (
        <div className="glass-panel p-4 animate-slide-up" style={{animationDelay: '0.1s'}}>
            <h2 className="text-xl font-bold mb-4">Risk Trend</h2>
            <div className="h-48">
                <canvas ref={chartRef}></canvas>
            </div>
        </div>
    );
}