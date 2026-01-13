/**
 * 3D Visualization for the Tiny U-Net model
 * Shows every weight as a cube, every activation as a colored plane
 * Layout: horizontal flow from input to output
 */

import * as THREE from 'three';
import {
  TinyUNet,
  TinyConvLayer,
  TimeEmbedding,
  ActivationTensor,
  ForwardPassState,
  TinyVisConfig,
  DEFAULT_TINY_VIS_CONFIG,
} from '../types/tiny';

// ============ HELPER FUNCTIONS ============

/**
 * Create a text sprite for labels (matching DiamondArchitectureVis style)
 */
function createTextSprite(
  text: string,
  size: number = 0.3,
  color: string = '#ffffff'
): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 256;
  const context = canvas.getContext('2d')!;

  context.fillStyle = 'transparent';
  context.fillRect(0, 0, canvas.width, canvas.height);

  // Better text rendering - bold Courier New
  context.font = 'bold 72px "Courier New", monospace';
  context.fillStyle = color;
  context.textAlign = 'center';
  context.textBaseline = 'middle';

  // Add subtle shadow for better readability
  context.shadowColor = 'rgba(0, 0, 0, 0.8)';
  context.shadowBlur = 8;
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
  sprite.scale.set(size * 8, size * 2, 1);

  return sprite;
}

/**
 * Map a value to a color (blue for positive, red for negative)
 * Brighter, more saturated colors for better visibility
 */
function weightToColor(value: number, config: TinyVisConfig): THREE.Color {
  const absVal = Math.min(Math.abs(value), 1);
  // Keep minimum brightness but allow stronger highlights for large magnitudes
  const intensity = Math.min(1.2, 0.55 + 0.8 * absVal);
  
  if (value >= 0) {
    // Positive: bright cyan-blue
    return new THREE.Color(config.positiveColor).multiplyScalar(intensity);
  } else {
    // Negative: bright red-orange
    return new THREE.Color(config.negativeColor).multiplyScalar(intensity);
  }
}

/**
 * Map activation value to color (viridis-like)
 */
function activationToColor(value: number): THREE.Color {
  // Clamp to [0, 1]
  const v = Math.max(0, Math.min(1, (value + 1) / 2)); // Assume values in [-1, 1]
  
  // Simple viridis approximation
  const r = Math.max(0, Math.min(1, 0.267 + 0.005 * v + 2.5 * v * v - 1.8 * v * v * v));
  const g = Math.max(0, Math.min(1, 0.004 + 1.4 * v - 0.4 * v * v));
  const b = Math.max(0, Math.min(1, 0.329 + 1.4 * v - 1.7 * v * v + 0.5 * v * v * v));
  
  return new THREE.Color(r, g, b);
}

// ============ WEIGHT VISUALIZATION ============

/**
 * Create a single weight cube
 */
function createWeightCube(
  value: number,
  config: TinyVisConfig
): THREE.Mesh {
  // Minimum size so all weights are visible
  const size = config.kernelScale * (0.6 + 0.4 * Math.min(Math.abs(value), 1));
  const geometry = new THREE.BoxGeometry(size, size, size);
  const color = weightToColor(value, config);
  
  const material = new THREE.MeshStandardMaterial({
    color,
    metalness: 0.1,
    roughness: 0.4,
    emissive: color,
    emissiveIntensity: 0.6,
  });
  
  return new THREE.Mesh(geometry, material);
}

/**
 * Create a 3×3 kernel visualization
 * Returns a group with 9 weight cubes arranged in a grid
 */
function createKernelVis(
  weights: number[][],
  config: TinyVisConfig
): THREE.Group {
  const group = new THREE.Group();
  const spacing = config.kernelScale * 1.2;
  
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const cube = createWeightCube(weights[i][j], config);
      cube.position.set(
        (j - 1) * spacing,
        (1 - i) * spacing, // Flip Y so [0][0] is top-left
        0
      );
      group.add(cube);
    }
  }
  
  return group;
}

/**
 * Create visualization for a conv layer
 * Shows all kernels arranged by output channel (X) and input channel (Z)
 */
function createConvLayerVis(
  layer: TinyConvLayer,
  config: TinyVisConfig
): THREE.Group {
  const group = new THREE.Group();
  
  const kernelWidth = config.kernelScale * 3 * 1.2;
  const ocSpacing = kernelWidth + 0.5;
  const icSpacing = config.channelSpacing;
  
  // Create kernels
  for (let oc = 0; oc < layer.outChannels; oc++) {
    for (let ic = 0; ic < layer.inChannels; ic++) {
      const kernel = createKernelVis(layer.kernels[oc][ic].weights, config);
      kernel.position.set(
        (oc - (layer.outChannels - 1) / 2) * ocSpacing,
        0,
        (ic - (layer.inChannels - 1) / 2) * icSpacing
      );
      group.add(kernel);
    }
  }
  
  // Bias visualization (small spheres below kernels)
  for (let oc = 0; oc < layer.outChannels; oc++) {
    const biasColor = weightToColor(layer.bias[oc], config);
    const biasGeo = new THREE.SphereGeometry(config.kernelScale * 0.4, 12, 12);
    const biasMat = new THREE.MeshStandardMaterial({
      color: biasColor,
      metalness: 0.1,
      roughness: 0.4,
      emissive: biasColor,
      emissiveIntensity: 0.6,
    });
    const biasMesh = new THREE.Mesh(biasGeo, biasMat);
    biasMesh.position.set(
      (oc - (layer.outChannels - 1) / 2) * ocSpacing,
      -kernelWidth * 0.7,
      0
    );
    group.add(biasMesh);
  }
  
  // Label
  const label = createTextSprite(layer.name, 0.35, '#00ffff');
  label.position.set(0, kernelWidth * 0.8, 0);
  group.add(label);
  
  // Channel info
  const channelLabel = createTextSprite(
    `${layer.inChannels}→${layer.outChannels}ch`,
    0.25,
    '#888888'
  );
  channelLabel.position.set(0, -kernelWidth * 1.1, 0);
  group.add(channelLabel);
  
  return group;
}

/**
 * Create time embedding visualization
 */
function createTimeEmbedVis(
  timeEmbed: TimeEmbedding,
  config: TinyVisConfig
): THREE.Group {
  const group = new THREE.Group();
  
  // Title
  const title = createTextSprite('Time Embed', 0.3, '#ffaa00');
  title.position.set(0, 1, 0);
  group.add(title);
  
  // Weights as vertical bars
  const barWidth = 0.3;
  const barSpacing = 0.5;
  
  for (let i = 0; i < timeEmbed.outputDim; i++) {
    const barColor = weightToColor(timeEmbed.weights[i], config);
    const height = Math.abs(timeEmbed.weights[i]) * 1.5 + 0.2;
    const geo = new THREE.BoxGeometry(barWidth, height, barWidth);
    const mat = new THREE.MeshStandardMaterial({
      color: barColor,
      metalness: 0.1,
      roughness: 0.4,
      emissive: barColor,
      emissiveIntensity: 0.6,
    });
    const bar = new THREE.Mesh(geo, mat);
    bar.position.set((i - (timeEmbed.outputDim - 1) / 2) * barSpacing, 0, 0);
    group.add(bar);
  }
  
  return group;
}

// ============ ACTIVATION VISUALIZATION ============

/**
 * Create visualization for an activation tensor
 * Each channel is a colored plane, stacked in Z
 */
function createActivationVis(
  activation: ActivationTensor,
  config: TinyVisConfig,
  label: string = ''
): THREE.Group {
  const group = new THREE.Group();
  
  const pixelSize = config.activationScale * 0.4;
  const pixelSpacing = pixelSize * 1.1;
  const channelSpacing = config.channelSpacing * 0.5;
  
  // First pass: find min and max values across all channels
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let c = 0; c < activation.channels; c++) {
    for (let h = 0; h < activation.height; h++) {
      for (let w = 0; w < activation.width; w++) {
        const value = activation.data[c][h][w];
        if (isFinite(value)) {
          minVal = Math.min(minVal, value);
          maxVal = Math.max(maxVal, value);
        }
      }
    }
  }
  
  // Ensure we have a valid range
  if (!isFinite(minVal) || !isFinite(maxVal) || minVal === maxVal) {
    minVal = 0;
    maxVal = 1;
  }
  const range = maxVal - minVal;
  
  for (let c = 0; c < activation.channels; c++) {
    const channelGroup = new THREE.Group();
    
    for (let h = 0; h < activation.height; h++) {
      for (let w = 0; w < activation.width; w++) {
        const value = activation.data[c][h][w];
        // Normalize to [0, 1] based on actual data range
        const normalized = (value - minVal) / range;
        // Full grayscale: 0 = black, 1 = white
        const color = new THREE.Color(normalized, normalized, normalized);
        
        const geo = new THREE.BoxGeometry(pixelSize, pixelSize, pixelSize * 0.3);
        const mat = new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.7,
        });
        
        const pixel = new THREE.Mesh(geo, mat);
        pixel.position.set(
          (w - (activation.width - 1) / 2) * pixelSpacing,
          ((activation.height - 1) / 2 - h) * pixelSpacing,
          0
        );
        channelGroup.add(pixel);
      }
    }
    
    channelGroup.position.z = (c - (activation.channels - 1) / 2) * channelSpacing;
    group.add(channelGroup);
  }
  
  // Label
  if (label) {
    const labelSprite = createTextSprite(label, 0.2, '#aaaaaa');
    labelSprite.position.set(0, -pixelSpacing * 1.5, 0);
    group.add(labelSprite);
  }
  
  return group;
}

// ============ DATA FLOW VISUALIZATION ============

/**
 * Create an arrow between two points
 */
function createArrow(
  from: THREE.Vector3,
  to: THREE.Vector3,
  color: number = 0xffffff
): THREE.Group {
  const group = new THREE.Group();
  
  const direction = to.clone().sub(from);
  const length = direction.length();
  direction.normalize();
  
  // Arrow shaft
  const shaftGeo = new THREE.CylinderGeometry(0.03, 0.03, length * 0.85, 8);
  const shaftMat = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.3,
  });
  const shaft = new THREE.Mesh(shaftGeo, shaftMat);
  
  // Arrow head
  const headGeo = new THREE.ConeGeometry(0.1, 0.2, 8);
  const head = new THREE.Mesh(headGeo, shaftMat);
  head.position.y = length * 0.85 / 2 + 0.1;
  
  group.add(shaft);
  group.add(head);
  
  // Orient arrow
  const midpoint = from.clone().add(to).multiplyScalar(0.5);
  group.position.copy(midpoint);
  group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  
  return group;
}

/**
 * Create a curved skip connection
 */
function createSkipConnection(
  from: THREE.Vector3,
  to: THREE.Vector3,
  height: number = 3
): THREE.Line {
  const curve = new THREE.QuadraticBezierCurve3(
    from,
    new THREE.Vector3((from.x + to.x) / 2, from.y + height, from.z),
    to
  );
  
  const points = curve.getPoints(50);
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineDashedMaterial({
    color: 0xff00ff,
    dashSize: 0.2,
    gapSize: 0.1,
    linewidth: 2,
  });
  
  const line = new THREE.Line(geometry, material);
  line.computeLineDistances();
  
  return line;
}

// ============ MAIN VISUALIZATION ============

/**
 * Create the complete tiny U-Net visualization
 */
export function createTinyUNetVisualization(
  model: TinyUNet,
  config: TinyVisConfig = DEFAULT_TINY_VIS_CONFIG
): THREE.Group {
  const group = new THREE.Group();
  const spacing = config.layerSpacing;
  
  // Layer positions (left to right flow)
  const positions = {
    input: -spacing * 2.5,
    inputConv: -spacing * 1.5,
    encoder: -spacing * 0.5,
    downsample: spacing * 0.2,
    bottleneck: spacing * 0.5,
    upsample: spacing * 0.8,
    skipConcat: spacing * 1.2,
    decoder: spacing * 1.8,
    output: spacing * 2.8,
  };
  
  // ===== TIME EMBEDDING (top) =====
  if (config.showTimeEmbedding) {
    const timeVis = createTimeEmbedVis(model.timeEmbed, config);
    timeVis.position.set(positions.inputConv, 4, 0);
    group.add(timeVis);
    
    // Arrow showing injection point
    const timeArrow = createArrow(
      new THREE.Vector3(positions.inputConv, 3, 0),
      new THREE.Vector3(positions.inputConv, 1.5, 0),
      0xffaa00
    );
    group.add(timeArrow);
  }
  
  // ===== INPUT CONV =====
  const inputConvVis = createConvLayerVis(model.inputConv, config);
  inputConvVis.position.set(positions.inputConv, 0, 0);
  group.add(inputConvVis);
  
  // ===== ENCODER CONV =====
  const encoderVis = createConvLayerVis(model.encoderConv, config);
  encoderVis.position.set(positions.encoder, 0, 0);
  group.add(encoderVis);
  
  // ===== DOWNSAMPLE INDICATOR =====
  const downLabel = createTextSprite('↓2×', 0.4, '#ff9900');
  downLabel.position.set(positions.downsample, 0, 0);
  group.add(downLabel);
  
  // ===== BOTTLENECK CONV =====
  const bottleneckVis = createConvLayerVis(model.bottleneckConv, config);
  bottleneckVis.position.set(positions.bottleneck, -2, 0); // Lower for U shape
  group.add(bottleneckVis);
  
  // ===== UPSAMPLE INDICATOR =====
  const upLabel = createTextSprite('↑2×', 0.4, '#00ff99');
  upLabel.position.set(positions.upsample, -1, 0);
  group.add(upLabel);
  
  // ===== SKIP CONCAT INDICATOR =====
  const concatLabel = createTextSprite('concat', 0.25, '#ff00ff');
  concatLabel.position.set(positions.skipConcat, 0.8, 0);
  group.add(concatLabel);
  
  // ===== DECODER CONV =====
  const decoderVis = createConvLayerVis(model.decoderConv, config);
  decoderVis.position.set(positions.decoder, 0, 0);
  group.add(decoderVis);
  
  // ===== OUTPUT CONV =====
  const outputVis = createConvLayerVis(model.outputConv, config);
  outputVis.position.set(positions.output, 0, 0);
  group.add(outputVis);
  
  // ===== FLOW ARROWS =====
  if (config.showDataFlow) {
    const arrowY = 0;
    const arrows = [
      { from: positions.inputConv + 1.5, to: positions.encoder - 1.5 },
      { from: positions.encoder + 1.5, to: positions.downsample - 0.5 },
      { from: positions.downsample + 0.5, to: positions.bottleneck - 1.5, y: -1 },
      { from: positions.bottleneck + 1.5, to: positions.upsample - 0.5, y: -1.5 },
      { from: positions.upsample + 0.5, to: positions.skipConcat - 0.5, y: -0.5 },
      { from: positions.skipConcat + 0.5, to: positions.decoder - 1.5 },
      { from: positions.decoder + 1.5, to: positions.output - 1.5 },
    ];
    
    for (const arrow of arrows) {
      const arrowVis = createArrow(
        new THREE.Vector3(arrow.from, arrow.y ?? arrowY, 0),
        new THREE.Vector3(arrow.to, arrow.y ?? arrowY, 0),
        0x666666
      );
      group.add(arrowVis);
    }
  }
  
  // ===== SKIP CONNECTION =====
  if (config.showSkipConnection) {
    const skipLine = createSkipConnection(
      new THREE.Vector3(positions.encoder + 1, 1, 0),
      new THREE.Vector3(positions.skipConcat, 0.5, 0),
      2.5
    );
    group.add(skipLine);
    
    const skipLabel = createTextSprite('skip', 0.2, '#ff00ff');
    skipLabel.position.set((positions.encoder + positions.skipConcat) / 2, 3.5, 0);
    group.add(skipLabel);
  }
  
  // ===== TITLE =====
  const title = createTextSprite('Tiny U-Net (~193 params)', 0.5, '#00ffff');
  title.position.set(0, 6, 0);
  group.add(title);
  
  return group;
}

/**
 * Create visualization with forward pass activations
 */
export function createTinyUNetWithActivations(
  model: TinyUNet,
  state: ForwardPassState,
  config: TinyVisConfig = DEFAULT_TINY_VIS_CONFIG
): THREE.Group {
  const group = createTinyUNetVisualization(model, config);
  
  if (!config.showActivations) return group;
  
  const spacing = config.layerSpacing;
  const actY = -4; // Below the weights
  
  // Position activations below each layer
  const positions = {
    input: -spacing * 2.5,
    afterInputConv: -spacing * 1.5,
    afterEncoder: -spacing * 0.5,
    afterDownsample: spacing * 0.2,
    afterBottleneck: spacing * 0.5,
    afterUpsample: spacing * 0.8,
    afterSkipConcat: spacing * 1.2,
    afterDecoder: spacing * 1.8,
    output: spacing * 2.8,
  };
  
  // Input
  const inputVis = createActivationVis(state.noisyInput, config, 'input');
  inputVis.position.set(positions.input, actY, 0);
  group.add(inputVis);
  
  // After input conv
  const afterInputVis = createActivationVis(state.afterInputConv, config, '');
  afterInputVis.position.set(positions.afterInputConv, actY, 0);
  group.add(afterInputVis);
  
  // After encoder
  const afterEncVis = createActivationVis(state.afterEncoder, config, '');
  afterEncVis.position.set(positions.afterEncoder, actY, 0);
  group.add(afterEncVis);
  
  // After downsample (smaller)
  const afterDownVis = createActivationVis(state.afterDownsample, config, '1×1');
  afterDownVis.position.set(positions.afterDownsample, actY - 2, 0);
  group.add(afterDownVis);
  
  // After bottleneck
  const afterBnVis = createActivationVis(state.afterBottleneck, config, '');
  afterBnVis.position.set(positions.afterBottleneck, actY - 2, 0);
  group.add(afterBnVis);
  
  // After upsample
  const afterUpVis = createActivationVis(state.afterUpsample, config, '');
  afterUpVis.position.set(positions.afterUpsample, actY - 1, 0);
  group.add(afterUpVis);
  
  // After skip concat (4 channels)
  const afterConcatVis = createActivationVis(state.afterSkipConcat, config, '4ch');
  afterConcatVis.position.set(positions.afterSkipConcat, actY, 0);
  group.add(afterConcatVis);
  
  // After decoder
  const afterDecVis = createActivationVis(state.afterDecoder, config, '');
  afterDecVis.position.set(positions.afterDecoder, actY, 0);
  group.add(afterDecVis);
  
  // Output
  const outputVis = createActivationVis(state.output, config, 'output');
  outputVis.position.set(positions.output, actY, 0);
  group.add(outputVis);
  
  // Timestep indicator
  const tLabel = createTextSprite(`t = ${state.timestep.toFixed(2)}`, 0.3, '#ffaa00');
  tLabel.position.set(positions.input, 4, 0);
  group.add(tLabel);
  
  return group;
}
