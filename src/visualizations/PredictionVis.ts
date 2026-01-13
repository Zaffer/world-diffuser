import * as THREE from "three";

export function createPredictionVisualization(predictions: number[][]): THREE.Group {
  const group = new THREE.Group();
  
  // Handle undefined or empty predictions data
  if (!predictions || predictions.length === 0) {
    console.warn("Empty or undefined predictions provided to createPredictionVisualization");
    return group; // Return empty group
  }
  
  const cellSize = 0.5;
  const spacing = 0.1;

  // Find min and max values for better color scaling
  let minValue = Number.MAX_VALUE;
  let maxValue = Number.MIN_VALUE;
  
  // First pass - get min/max while handling NaN values
  predictions.forEach(row => {
    row.forEach(value => {
      // Skip NaN or infinite values
      if (!isNaN(value) && isFinite(value)) {
        minValue = Math.min(minValue, value);
        maxValue = Math.max(maxValue, value);
      }
    });
  });
  
  // If we have invalid data, use defaults
  if (minValue === Number.MAX_VALUE || maxValue === Number.MIN_VALUE || minValue === maxValue) {
    console.warn("Invalid prediction data range, using defaults");
    minValue = 0;
    maxValue = 5;
  }

  predictions.forEach((row, i) => {
    row.forEach((value, j) => {
      const geometry = new THREE.PlaneGeometry(cellSize, cellSize);
      
      // Replace NaN with default value for visualization purposes
      const safeValue = isNaN(value) || !isFinite(value) ? (minValue + maxValue) / 2 : value;
      
      // Force a minimum brightness difference between colors
      const range = Math.max(1, maxValue - minValue);
      const normalizedValue = Math.max(0, Math.min(1, (safeValue - minValue) / range));
      
      // Full contrast grayscale: 0 = total black, 1 = total white
      const grayValue = normalizedValue;
      
      const material = new THREE.MeshStandardMaterial({ 
        color: new THREE.Color(grayValue, grayValue, grayValue), 
        side: THREE.DoubleSide, 
        emissive: new THREE.Color(grayValue, grayValue, grayValue),
        emissiveIntensity: 0.8,
        metalness: 0.0,
        roughness: 0.8,
      });
      
      const cell = new THREE.Mesh(geometry, material);

      cell.position.set(
        j * (cellSize + spacing) - (predictions[0].length * (cellSize + spacing)) / 2 + cellSize / 2,
        -i * (cellSize + spacing) + (predictions.length * (cellSize + spacing)) / 2 - cellSize / 2,
        0
      );

      group.add(cell);
    });
  });

  return group;
}