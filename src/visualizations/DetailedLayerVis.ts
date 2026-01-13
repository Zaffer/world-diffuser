/**
 * Detailed layer visualization showing weights and activations
 * This is shown when drilling down into a specific block
 */

import * as THREE from 'three';
import {
  ConvLayer,
  ResidualBlock,
  FeatureMap,
} from '../types/diamond';

/**
 * Create a detailed visualization of convolutional kernels
 * Shows a grid of kernel weights with color coding
 */
export function createKernelGridVisualization(
  convLayer: ConvLayer,
  maxKernels: number = 16
): THREE.Group {
  const group = new THREE.Group();

  const displayOutChannels = Math.min(convLayer.outChannels, 4);
  const displayInChannels = Math.min(convLayer.inChannels, 4);

  const kernelSize = 0.8;
  const kernelSpacing = 1.2;

  // Create grid of kernels
  for (let oc = 0; oc < displayOutChannels; oc++) {
    for (let ic = 0; ic < displayInChannels; ic++) {
      const kernel = convLayer.weights[oc][ic];
      const kernelVis = createSingleKernel(kernel, kernelSize);

      const x = ic * kernelSpacing - (displayInChannels * kernelSpacing) / 2;
      const y = -oc * kernelSpacing + (displayOutChannels * kernelSpacing) / 2;

      kernelVis.position.set(x, y, 0);
      group.add(kernelVis);

      // Add label
      const label = createSmallLabel(
        `[${oc},${ic}]`,
        new THREE.Vector3(x, y - kernelSize * 0.7, 0),
        0.15
      );
      group.add(label);
    }
  }

  // Add axis labels
  const inChannelLabel = createSmallLabel(
    `In: ${convLayer.inChannels}ch`,
    new THREE.Vector3(0, (displayOutChannels * kernelSpacing) / 2 + 0.5, 0),
    0.2,
    '#00aaff'
  );
  group.add(inChannelLabel);

  const outChannelLabel = createSmallLabel(
    `Out: ${convLayer.outChannels}ch`,
    new THREE.Vector3(-(displayInChannels * kernelSpacing) / 2 - 0.8, 0, 0),
    0.2,
    '#00ff88'
  );
  group.add(outChannelLabel);

  return group;
}

/**
 * Create a single kernel visualization (3x3 or 1x1)
 */
function createSingleKernel(
  kernel: number[][],
  size: number
): THREE.Group {
  const group = new THREE.Group();

  const kh = kernel.length;
  const kw = kernel[0].length;
  const cellSize = size / Math.max(kh, kw);
  const spacing = cellSize * 1.05;

  for (let h = 0; h < kh; h++) {
    for (let w = 0; w < kw; w++) {
      const weight = kernel[h][w];

      // Color based on weight value
      const absWeight = Math.abs(weight);
      const normalizedWeight = Math.tanh(absWeight * 2); // Compress large values

      let color: THREE.Color;
      if (weight > 0.01) {
        // Positive weights: green
        color = new THREE.Color(0x00ff00).multiplyScalar(0.3 + 0.7 * normalizedWeight);
      } else if (weight < -0.01) {
        // Negative weights: red
        color = new THREE.Color(0xff0000).multiplyScalar(0.3 + 0.7 * normalizedWeight);
      } else {
        // Near-zero weights: gray
        color = new THREE.Color(0x444444);
      }

      const geometry = new THREE.BoxGeometry(cellSize * 0.9, cellSize * 0.9, cellSize * 0.3);
      const material = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: 0.8,
      });
      const cube = new THREE.Mesh(geometry, material);

      const x = w * spacing - (kw * spacing) / 2 + spacing / 2;
      const y = -h * spacing + (kh * spacing) / 2 - spacing / 2;
      cube.position.set(x, y, 0);

      // Store weight value for interaction
      cube.userData = {
        weight: weight,
        position: [h, w],
      };

      group.add(cube);
    }
  }

  return group;
}

/**
 * Create a feature map visualization (activations)
 * Shows spatial activation patterns as a heatmap
 */
export function createFeatureMapVisualization(
  featureMap: FeatureMap,
  channelIndex: number = 0
): THREE.Group {
  const group = new THREE.Group();

  const { channels, height, width, data } = featureMap;

  // Clamp channel index
  const channel = Math.min(channelIndex, channels - 1);
  const channelData = data[channel];

  const cellSize = 0.15;
  const spacing = cellSize * 1.1;

  // Find min/max for normalization
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      const val = channelData[h][w];
      minVal = Math.min(minVal, val);
      maxVal = Math.max(maxVal, val);
    }
  }

  const range = maxVal - minVal || 1;

  // Create heatmap
  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      const value = channelData[h][w];
      const normalized = (value - minVal) / range;

      // Heatmap color: blue (low) → cyan → green → yellow → red (high)
      const color = new THREE.Color();
      if (normalized < 0.25) {
        color.setRGB(0, 0, 0.5 + normalized * 2);
      } else if (normalized < 0.5) {
        color.setRGB(0, (normalized - 0.25) * 4, 1);
      } else if (normalized < 0.75) {
        color.setRGB((normalized - 0.5) * 4, 1, 1 - (normalized - 0.5) * 4);
      } else {
        color.setRGB(1, 1 - (normalized - 0.75) * 4, 0);
      }

      const geometry = new THREE.BoxGeometry(cellSize, cellSize, cellSize * 0.5);
      const material = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: 0.9,
      });
      const cube = new THREE.Mesh(geometry, material);

      const x = w * spacing - (width * spacing) / 2 + spacing / 2;
      const y = -h * spacing + (height * spacing) / 2 - spacing / 2;
      cube.position.set(x, y, 0);

      cube.userData = {
        activation: value,
        position: [h, w],
        channel: channel,
      };

      group.add(cube);
    }
  }

  // Add info label
  const infoLabel = createSmallLabel(
    `Channel ${channel}/${channels} [${height}×${width}]`,
    new THREE.Vector3(0, (height * spacing) / 2 + 0.4, 0),
    0.25,
    '#ffff00'
  );
  group.add(infoLabel);

  // Add colorbar legend
  const legendY = -(height * spacing) / 2 - 0.6;
  const legendWidth = width * spacing;
  const legendHeight = 0.3;
  const legendSteps = 20;

  for (let i = 0; i < legendSteps; i++) {
    const normalized = i / (legendSteps - 1);
    const color = new THREE.Color();

    if (normalized < 0.25) {
      color.setRGB(0, 0, 0.5 + normalized * 2);
    } else if (normalized < 0.5) {
      color.setRGB(0, (normalized - 0.25) * 4, 1);
    } else if (normalized < 0.75) {
      color.setRGB((normalized - 0.5) * 4, 1, 1 - (normalized - 0.5) * 4);
    } else {
      color.setRGB(1, 1 - (normalized - 0.75) * 4, 0);
    }

    const geometry = new THREE.BoxGeometry(
      legendWidth / legendSteps,
      legendHeight,
      0.05
    );
    const material = new THREE.MeshBasicMaterial({ color: color });
    const bar = new THREE.Mesh(geometry, material);

    const x = (i / legendSteps) * legendWidth - legendWidth / 2;
    bar.position.set(x, legendY, 0);
    group.add(bar);
  }

  // Min/max labels
  const minLabel = createSmallLabel(
    minVal.toFixed(2),
    new THREE.Vector3(-legendWidth / 2, legendY - 0.3, 0),
    0.15
  );
  group.add(minLabel);

  const maxLabel = createSmallLabel(
    maxVal.toFixed(2),
    new THREE.Vector3(legendWidth / 2, legendY - 0.3, 0),
    0.15
  );
  group.add(maxLabel);

  return group;
}

/**
 * Create a residual block detailed view
 */
export function createResidualBlockDetailView(
  block: ResidualBlock
): THREE.Group {
  const group = new THREE.Group();

  const spacing = 3;

  // Conv1 kernels
  const conv1Vis = createKernelGridVisualization(block.conv1, 16);
  conv1Vis.position.set(-spacing, 1, 0);
  group.add(conv1Vis);

  const conv1Label = createSmallLabel(
    'Conv1 Weights',
    new THREE.Vector3(-spacing, -1.5, 0),
    0.3,
    '#00aaff'
  );
  group.add(conv1Label);

  // Conv2 kernels
  const conv2Vis = createKernelGridVisualization(block.conv2, 16);
  conv2Vis.position.set(spacing, 1, 0);
  group.add(conv2Vis);

  const conv2Label = createSmallLabel(
    'Conv2 Weights',
    new THREE.Vector3(spacing, -1.5, 0),
    0.3,
    '#00ff88'
  );
  group.add(conv2Label);

  // Block info
  const infoLabel = createSmallLabel(
    'Residual Block Details',
    new THREE.Vector3(0, 3, 0),
    0.4,
    '#ffffff'
  );
  group.add(infoLabel);

  return group;
}

/**
 * Create animated data flow particles
 */
export function createDataFlowParticles(
  fromPos: THREE.Vector3,
  toPos: THREE.Vector3,
  count: number = 20
): THREE.Group {
  const group = new THREE.Group();

  // Create path
  const curve = new THREE.LineCurve3(fromPos, toPos);

  for (let i = 0; i < count; i++) {
    const geometry = new THREE.SphereGeometry(0.05, 8, 8);
    const material = new THREE.MeshBasicMaterial({
      color: 0x00ffff,
      transparent: true,
      opacity: 0.8,
    });
    const particle = new THREE.Mesh(geometry, material);

    // Store animation data
    particle.userData = {
      curve: curve,
      progress: i / count,
      speed: 0.01 + Math.random() * 0.02,
    };

    group.add(particle);
  }

  return group;
}

/**
 * Update data flow particles animation
 */
export function updateDataFlowParticles(group: THREE.Group, deltaTime: number): void {
  group.children.forEach((child) => {
    if (child.userData.curve) {
      const { curve, speed } = child.userData;

      // Update progress
      child.userData.progress = (child.userData.progress + speed * deltaTime) % 1.0;

      // Update position along curve
      const point = curve.getPoint(child.userData.progress);
      child.position.copy(point);

      // Fade in/out at start/end
      const material = (child as THREE.Mesh).material as THREE.Material;
      if (child.userData.progress < 0.1) {
        material.opacity = child.userData.progress / 0.1;
      } else if (child.userData.progress > 0.9) {
        material.opacity = (1.0 - child.userData.progress) / 0.1;
      } else {
        material.opacity = 0.8;
      }
    }
  });
}

/**
 * Create a small text label
 */
function createSmallLabel(
  text: string,
  position: THREE.Vector3,
  size: number,
  color: string = '#cccccc'
): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 128;
  const context = canvas.getContext('2d')!;

  context.fillStyle = 'transparent';
  context.fillRect(0, 0, canvas.width, canvas.height);

  context.font = 'bold 48px monospace';
  context.fillStyle = color;
  context.textAlign = 'center';
  context.textBaseline = 'middle';

  context.shadowColor = 'rgba(0, 0, 0, 0.8)';
  context.shadowBlur = 4;
  context.shadowOffsetX = 2;
  context.shadowOffsetY = 2;

  context.fillText(text, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
  });
  const sprite = new THREE.Sprite(material);

  sprite.scale.set(size * 4, size, 1);
  sprite.position.copy(position);

  return sprite;
}
