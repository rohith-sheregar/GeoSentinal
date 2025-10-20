// This file is not strictly necessary for the current implementation as the logic is in MapContainer.js
// However, it is good practice to keep utility functions separate.

export function getBoundingBox(lat, lng) {
    const lat_offset = 2.5 / 111.1; 
    const lng_offset = 2.5 / (111.1 * Math.cos(lat * Math.PI / 180));

    return [
        [lat - lat_offset, lng - lng_offset],
        [lat - lat_offset, lng + lng_offset],
        [lat + lat_offset, lng + lng_offset],
        [lat + lat_offset, lng - lng_offset]
    ];
}