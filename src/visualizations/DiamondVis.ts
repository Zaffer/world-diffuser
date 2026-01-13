/**
 * Visualization for DIAMOND diffusion world model architecture
 * Shows the mechanistic details of U-Net components
 */

import * as THREE from 'three';
import {
  ConvLayer,
  LayerVisualizationConfig,
  DEFAULT_LAYER_VIS_CONFIG,
  ResidualBlock,
  DownsampleLayer,
  UpsampleLayer,
  SkipConnection
} from '../types/diamond';

/**
 * Visualize a single convolutional kernel as a 3D grid
 * Each weight value is represented as a colored cube
 */
export function createConvKernelVisualization(
  kernel: number[][], // [height][width]
  position: THREE.Vector3,
  config: LayerVisualizationConfig = DEFAULT_LAYER_VIS_CONFIG
): THREE.Group {
  const group = new THREE.Group();
  const cubeSize = 0.1;
  const spacing = 0.12;

  const height = kernel.length;
  const width = kernel[0].length;

  // Color scheme for weights
  const negativeColor = new THREE.Color(0xff0000); // Red for negative
  const positiveColor = new THREE.Color(0x00ff00); // Green for positive
  const zeroColor = new THREE.Color(0x888888); // Gray for near-zero

  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      const weight = kernel[h][w] * config.weightScale;

      // Determine color based on weight value
      let color: THREE.Color;
      if (Math.abs(weight) < 0.01) {
        color = zeroColor.clone();
      } else if (weight > 0) {
        const intensity = Math.min(Math.abs(weight), 1.0);
        color = positiveColor.clone().multiplyScalar(intensity);
      } else {
        const intensity = Math.min(Math.abs(weight), 1.0);
        color = negativeColor.clone().multiplyScalar(intensity);
      }

      // Size based on magnitude
      const magnitude = Math.min(Math.abs(weight), 1.0);
      const size = cubeSize * (0.3 + 0.7 * magnitude);

      const geometry = new THREE.BoxGeometry(size, size, size);
      const material = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: 0.8,
      });
      const cube = new THREE.Mesh(geometry, material);

      // Position in a grid
      cube.position.set(
        position.x + (w - width / 2) * spacing,
        position.y + (h - height / 2) * spacing,
        position.z
      );

      // Store weight value in userData for interaction
      cube.userData = {
        type: 'conv_weight',
        value: weight,
        position: { h, w },
      };

      group.add(cube);
    }
  }

  return group;
}

/**
 * Visualize all kernels in a convolutional layer
 * Shows [outChannels][inChannels] grid of kernels
 */
export function createConvLayerVisualization(
  convLayer: ConvLayer,
  config: LayerVisualizationConfig = DEFAULT_LAYER_VIS_CONFIG
): THREE.Group {
  const group = new THREE.Group();
  const kernelSpacing = 1.5;
  const channelSpacing = 2.0;

  // For now, visualize only the first few output channels to avoid clutter
  const maxOutChannels = Math.min(convLayer.outChannels, 4);
  const maxInChannels = Math.min(convLayer.inChannels, 4);

  for (let oc = 0; oc < maxOutChannels; oc++) {
    for (let ic = 0; ic < maxInChannels; ic++) {
      const kernel = convLayer.weights[oc][ic];

      // Position kernels in a grid
      const position = new THREE.Vector3(
        ic * kernelSpacing,
        -oc * channelSpacing,
        0
      );

      const kernelVis = createConvKernelVisualization(kernel, position, config);
      group.add(kernelVis);

      // Add label for this kernel
      if (config.showWeights) {
        const label = createTextLabel(
          `K[${oc},${ic}]`,
          new THREE.Vector3(position.x, position.y - 0.5, position.z)
        );
        group.add(label);
      }
    }
  }

  // Add layer info label
  const layerInfo = `Conv2D: ${convLayer.inChannels}→${convLayer.outChannels}, ${convLayer.kernelSize}x${convLayer.kernelSize}`;
  const infoLabel = createTextLabel(layerInfo, new THREE.Vector3(0, 1.5, 0), 0.15);
  group.add(infoLabel);

  return group;
}

/**
 * Create a simple text label using a sprite
 * This is a placeholder - in a full implementation we'd use TextGeometry or canvas textures
 */
function createTextLabel(text: string, position: THREE.Vector3, size: number = 0.1): THREE.Sprite {
  // For now, create a simple colored sprite as a placeholder
  // TODO: Replace with actual text rendering
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const context = canvas.getContext('2d')!;

  context.fillStyle = '#222222';
  context.fillRect(0, 0, canvas.width, canvas.height);

  context.font = '20px monospace';
  context.fillStyle = '#ffffff';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(text, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture });
  const sprite = new THREE.Sprite(material);

  sprite.scale.set(size * 4, size, 1);
  sprite.position.copy(position);

  return sprite;
}

/**
 * Create a complete visualization of a minimal DIAMOND model
 * Start with just the first convolutional layer
 */
export function createMinimalDiamondVisualization(
  convLayer: ConvLayer,
  config: LayerVisualizationConfig = DEFAULT_LAYER_VIS_CONFIG
): THREE.Group {
  const group = new THREE.Group();

  // Add the convolution layer visualization
  const convVis = createConvLayerVisualization(convLayer, config);
  group.add(convVis);

  // Center the visualization
  group.position.set(0, 0, -2);

  return group;
}

/**
 * Create a wireframe container box for a block
 */
export function createBlockContainer(
  blockName: string,
  position: THREE.Vector3,
  size: THREE.Vector3
): THREE.Group {
  const group = new THREE.Group();

  // Wireframe box
  const geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
  const edges = new THREE.EdgesGeometry(geometry);
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x666666, linewidth: 1 });
  const wireframe = new THREE.LineSegments(edges, lineMaterial);
  group.add(wireframe);

  // Label
  const label = createTextLabel(blockName, new THREE.Vector3(0, size.y / 2 + 0.3, 0), 0.12);
  group.add(label);

  group.position.copy(position);
  return group;
}

/**
 * Visualize a residual block with all its components
 */
export function createResidualBlockVis(
  residualBlock: ResidualBlock,
  blockName: string,
  position: THREE.Vector3,
  config: LayerVisualizationConfig = DEFAULT_LAYER_VIS_CONFIG
): THREE.Group {
  const group = new THREE.Group();

  // Stack layers vertically within the block
  const layerSpacing = 0.8;
  let currentZ = 0;

  // Conv1
  const conv1Vis = createConvLayerVisualization(residualBlock.conv1, config);
  conv1Vis.position.z = currentZ;
  conv1Vis.scale.setScalar(0.3); // Scale down to fit in block
  group.add(conv1Vis);
  currentZ -= layerSpacing;

  // Conv2
  const conv2Vis = createConvLayerVisualization(residualBlock.conv2, config);
  conv2Vis.position.z = currentZ;
  conv2Vis.scale.setScalar(0.3);
  group.add(conv2Vis);

  // Add block container
  const containerSize = new THREE.Vector3(2, 1.5, 2);
  const container = createBlockContainer(blockName, new THREE.Vector3(0, 0, -layerSpacing / 2), containerSize);
  group.add(container);

  group.position.copy(position);
  return group;
}

/**
 * Visualize a downsampling operation
 */
export function createDownsampleVis(
  downsample: DownsampleLayer,
  position: THREE.Vector3
): THREE.Group {
  const group = new THREE.Group();

  // Arrow pointing downward
  const arrowGeometry = new THREE.ConeGeometry(0.15, 0.3, 8);
  const arrowMaterial = new THREE.MeshBasicMaterial({ color: 0xff9900 });
  const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
  arrow.rotation.x = Math.PI; // Point downward
  group.add(arrow);

  // Label showing spatial reduction
  const label = createTextLabel(`↓ ×${downsample.factor}`, new THREE.Vector3(0.5, 0, 0), 0.1);
  group.add(label);

  group.position.copy(position);
  return group;
}

/**
 * Visualize an upsampling operation
 */
export function createUpsampleVis(
  upsample: UpsampleLayer,
  position: THREE.Vector3
): THREE.Group {
  const group = new THREE.Group();

  // Arrow pointing upward
  const arrowGeometry = new THREE.ConeGeometry(0.15, 0.3, 8);
  const arrowMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff99 });
  const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
  group.add(arrow);

  // Label showing spatial expansion
  const label = createTextLabel(`↑ ×${upsample.factor}`, new THREE.Vector3(0.5, 0, 0), 0.1);
  group.add(label);

  group.position.copy(position);
  return group;
}

/**
 * Visualize a skip connection as a curved line
 */
export function createSkipConnectionVis(
  skipConnection: SkipConnection,
  fromPos: THREE.Vector3,
  toPos: THREE.Vector3
): THREE.Group {
  const group = new THREE.Group();

  // Create a curved path from encoder to decoder
  const curve = new THREE.CubicBezierCurve3(
    fromPos,
    new THREE.Vector3(fromPos.x + 2, fromPos.y, fromPos.z), // Control point 1
    new THREE.Vector3(toPos.x + 2, toPos.y, toPos.z),       // Control point 2
    toPos
  );

  const points = curve.getPoints(50);
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color: 0x9933ff, linewidth: 2 });
  const line = new THREE.Line(geometry, material);
  group.add(line);

  // Label at midpoint
  const midpoint = curve.getPoint(0.5);
  const shapeLabel = `[${skipConnection.fromShape.join(',')}]`;
  const label = createTextLabel(shapeLabel, midpoint, 0.08);
  group.add(label);

  return group;
}
