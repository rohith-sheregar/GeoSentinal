const { useEffect, useRef, useState } = React;

function MapContainer({ bounds }) {
    const mapRef = useRef(null);
    const [map, setMap] = useState(null);
    const [selectedArea, setSelectedArea] = useState(null);
    const [heatmap, setHeatmap] = useState(null);

    useEffect(() => {
        if (mapRef.current && !map) {
            const center = [(bounds.south + bounds.north) / 2, (bounds.west + bounds.east) / 2];
            const leafletMap = L.map(mapRef.current, {
                center: center,
                zoom: 13,
                minZoom: 3,
                maxZoom: 18
            });

            L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            }).addTo(leafletMap);

            setMap(leafletMap);
        }
    }, [mapRef, map, bounds]);

    useEffect(() => {
        if (map) {
            map.on('click', async (e) => {
                const { lat, lng } = e.latlng;

                if (selectedArea) {
                    map.removeLayer(selectedArea);
                }
                if (heatmap) {
                    map.removeLayer(heatmap);
                }

                const lat_offset = 2.5 / 111.1;
                const lng_offset = 2.5 / (111.1 * Math.cos(lat * Math.PI / 180));

                const polygonCoords = [
                    [lat - lat_offset, lng - lng_offset],
                    [lat - lat_offset, lng + lng_offset],
                    [lat + lat_offset, lng + lng_offset],
                    [lat + lat_offset, lng - lng_offset]
                ];

                const polygon = L.polygon(polygonCoords, { color: 'cyan' }).addTo(map);
                setSelectedArea(polygon);

                const polygonBounds = polygon.getBounds();
                const imageUrl = await fetch('http://127.0.0.1:5000/polygon_heatmap', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ coordinates: polygon.getLatLngs() })
                }).then(res => res.blob()).then(blob => URL.createObjectURL(blob));

                const imageOverlay = L.imageOverlay(imageUrl, polygonBounds, {
                    opacity: 0.7,
                    interactive: false
                }).addTo(map);
                setHeatmap(imageOverlay);

                const metrics = await fetch('http://127.0.0.1:5000/polygon_metrics', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ coordinates: polygon.getLatLngs() })
                }).then(res => res.json());

                window.dispatchEvent(new CustomEvent('analysis-update', { detail: metrics }));
                window.dispatchEvent(new CustomEvent('alert', { detail: metrics.explanation }));
            });
        }
        return () => {
            if (map) {
                map.off('click');
            }
        };
    }, [map, selectedArea, heatmap]);

    return <div id="map" ref={mapRef} className="w-full h-full rounded-2xl shadow-lg"></div>;
}