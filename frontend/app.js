const { useState, useEffect } = React;

function App() {
    const [bounds, setBounds] = useState(null);

    useEffect(() => {
        async function getBounds() {
            const b = await fetch('http://127.0.0.1:5000/bounds').then(res => res.json());
            setBounds(b);
        }
        getBounds();
    }, []);

    if (!bounds) {
        return <div>Loading...</div>;
    }

    return (
        <div className="h-screen w-screen bg-gray-900 text-white flex flex-col">
            <header className="bg-gray-800 shadow-md p-4 flex items-center justify-between">
                <h1 className="text-2xl font-bold">GeoSentinal</h1>
            </header>
            <main className="flex-grow flex p-4 gap-4">
                <div className="w-2/3 h-full">
                    {bounds && <MapContainer bounds={bounds} />}
                </div>
                <div className="w-1/3 h-full flex flex-col gap-4">
                    <AnalysisPanel />
                    <RiskChart />
                </div>
            </main>
            <AlertSystem />
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root'));