/**
 * Full U-Net architecture visualization for DIAMOND
 * Redesigned with weights visible inline in each block
 */

import * as THREE from 'three';
import {
  MinimalUNet,
  ArchitectureVisConfig,
  DEFAULT_ARCH_VIS_CONFIG,
  ConvLayer,
  ResidualBlock,
} from '../types/diamond';

/**
 * Create the complete minimal U-Net architecture visualization
 * Horizontal U-shape layout with inline weight visualizations
 */
export function createMinimalUNetVisualization(
  model: MinimalUNet,
  config: ArchitectureVisConfig = DEFAULT_ARCH_VIS_CONFIG
): THREE.Group {
  const group = new THREE.Group();

  // Base unit for all spacing
  const UNIT = 4;

  // ========== LAYOUT POSITIONS ==========
  const positions = {
    input: new THREE.Vector3(-UNIT * 3.5, UNIT * 3, 0),
    encoder: new THREE.Vector3(-UNIT * 3.5, 0, 0),
    downsample: new THREE.Vector3(-UNIT * 3.5, -UNIT * 1.5, 0),
    bottleneck: new THREE.Vector3(0, -UNIT * 3, 0),
    upsample: new THREE.Vector3(UNIT * 3.5, -UNIT * 1.5, 0),
    decoder: new THREE.Vector3(UNIT * 3.5, 0, 0),
    output: new THREE.Vector3(UNIT * 3.5, UNIT * 3, 0),
  };

  // ========== INPUT CONVOLUTION ==========
  const inputBlock = createConvBlockWithWeights(
    'Input Conv',
    '3→8 ch',
    model.inputConv,
    '#4488ff'
  );
  inputBlock.position.copy(positions.input);
  group.add(inputBlock);

  // ========== ENCODER STAGE ==========
  const encoderBlock = createResidualBlockWithWeights(
    'Encoder',
    model.encoder.blocks[0],
    '#00aaff'
  );
  encoderBlock.position.copy(positions.encoder);
  group.add(encoderBlock);

  // Downsample indicator
  const downsampleBlock = createOperationBlock('↓ ×2', '#ff9900');
  downsampleBlock.position.copy(positions.downsample);
  group.add(downsampleBlock);

  // ========== BOTTLENECK STAGE ==========
  const bottleneckBlock = createResidualBlockWithWeights(
    'Bottleneck',
    model.bottleneck,
    '#aa00ff'
  );
  bottleneckBlock.position.copy(positions.bottleneck);
  group.add(bottleneckBlock);

  // Upsample indicator
  const upsampleBlock = createOperationBlock('↑ ×2', '#00ff99');
  upsampleBlock.position.copy(positions.upsample);
  group.add(upsampleBlock);

  // ========== DECODER STAGE ==========
  const decoderBlock = createResidualBlockWithWeights(
    'Decoder',
    model.decoder.blocks[0],
    '#00ff88'
  );
  decoderBlock.position.copy(positions.decoder);
  group.add(decoderBlock);

  // ========== OUTPUT CONVOLUTION ==========
  const outputBlock = createConvBlockWithWeights(
    'Output Conv',
    '8→3 ch',
    model.outputConv,
    '#44ff88'
  );
  outputBlock.position.copy(positions.output);
  group.add(outputBlock);

  // ========== SKIP CONNECTION ==========
  if (config.showSkipConnections) {
    const skipCurve = new THREE.QuadraticBezierCurve3(
      positions.encoder.clone().add(new THREE.Vector3(2, 0, 0)),
      new THREE.Vector3(0, UNIT * 1.5, 0),
      positions.decoder.clone().add(new THREE.Vector3(-2, 0, 0))
    );

    const skipPoints = skipCurve.getPoints(50);
    const skipGeometry = new THREE.BufferGeometry().setFromPoints(skipPoints);
    const skipMaterial = new THREE.LineBasicMaterial({
      color: 0xff00ff,
      linewidth: 3,
      transparent: true,
      opacity: 0.6
    });
    const skipLine = new THREE.Line(skipGeometry, skipMaterial);
    group.add(skipLine);

    // Skip connection label
    const skipLabel = createLabel(
      'Skip Connection',
      new THREE.Vector3(0, UNIT * 1.8, 0),
      0.35,
      '#ff00ff'
    );
    group.add(skipLabel);
  }

  // ========== FLOW ARROWS ==========
  const arrowColor = 0xffffff;
  const arrowOffset = 2;

  // Input → Encoder
  createFlowArrow(
    group,
    positions.input.clone().add(new THREE.Vector3(0, -arrowOffset, 0)),
    positions.encoder.clone().add(new THREE.Vector3(0, arrowOffset, 0)),
    arrowColor
  );

  // Encoder → Downsample
  createFlowArrow(
    group,
    positions.encoder.clone().add(new THREE.Vector3(0, -arrowOffset, 0)),
    positions.downsample.clone().add(new THREE.Vector3(0, 0.3, 0)),
    arrowColor
  );

  // Downsample → Bottleneck
  createFlowArrow(
    group,
    positions.downsample.clone().add(new THREE.Vector3(0.5, -0.3, 0)),
    positions.bottleneck.clone().add(new THREE.Vector3(-2, 0.5, 0)),
    arrowColor
  );

  // Bottleneck → Upsample
  createFlowArrow(
    group,
    positions.bottleneck.clone().add(new THREE.Vector3(2, 0.5, 0)),
    positions.upsample.clone().add(new THREE.Vector3(-0.5, -0.3, 0)),
    arrowColor
  );

  // Upsample → Decoder
  createFlowArrow(
    group,
    positions.upsample.clone().add(new THREE.Vector3(0, 0.3, 0)),
    positions.decoder.clone().add(new THREE.Vector3(0, -arrowOffset, 0)),
    arrowColor
  );

  // Decoder → Output
  createFlowArrow(
    group,
    positions.decoder.clone().add(new THREE.Vector3(0, arrowOffset, 0)),
    positions.output.clone().add(new THREE.Vector3(0, -arrowOffset, 0)),
    arrowColor
  );

  return group;
}

/**
 * Create a residual block with inline weight visualization
 */
function createResidualBlockWithWeights(
  title: string,
  block: ResidualBlock,
  color: string
): THREE.Group {
  const group = new THREE.Group();

  // Title at top
  const titleLabel = createLabel(title, new THREE.Vector3(0, 2.5, 0), 0.4, color);
  group.add(titleLabel);

  // Conv1 on left
  const conv1Group = createCompactKernelGrid(block.conv1, 0.6);
  conv1Group.position.set(-1.5, 0.5, 0);
  group.add(conv1Group);

  const conv1Label = createLabel('Conv1', new THREE.Vector3(-1.5, -1.2, 0), 0.25, '#cccccc');
  group.add(conv1Label);

  // Conv2 on right
  const conv2Group = createCompactKernelGrid(block.conv2, 0.6);
  conv2Group.position.set(1.5, 0.5, 0);
  group.add(conv2Group);

  const conv2Label = createLabel('Conv2', new THREE.Vector3(1.5, -1.2, 0), 0.25, '#cccccc');
  group.add(conv2Label);

  // Flow arrow between Conv1 and Conv2
  createSmallFlowArrow(
    group,
    new THREE.Vector3(-0.6, 0.5, 0),
    new THREE.Vector3(0.6, 0.5, 0),
    0xcccccc
  );

  // Residual connection arc
  const residualCurve = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(-1.5, 1.5, 0),
    new THREE.Vector3(0, 2, 0),
    new THREE.Vector3(1.5, 1.5, 0)
  );

  const residualPoints = residualCurve.getPoints(20);
  const residualGeometry = new THREE.BufferGeometry().setFromPoints(residualPoints);
  const residualMaterial = new THREE.LineDashedMaterial({
    color: 0xffaa00,
    linewidth: 2,
    dashSize: 0.1,
    gapSize: 0.1,
  });
  const residualLine = new THREE.Line(residualGeometry, residualMaterial);
  residualLine.computeLineDistances();
  group.add(residualLine);

  const residualLabel = createLabel('Residual', new THREE.Vector3(0, 2.2, 0), 0.2, '#ffaa00');
  group.add(residualLabel);

  // Bounding box for the whole block
  const boxGeometry = new THREE.BoxGeometry(4, 3.5, 1);
  const boxEdges = new THREE.EdgesGeometry(boxGeometry);
  const boxMaterial = new THREE.LineBasicMaterial({
    color: color,
    linewidth: 2,
    transparent: true,
    opacity: 0.5,
  });
  const boxWireframe = new THREE.LineSegments(boxEdges, boxMaterial);
  boxWireframe.position.y = 0.2;
  group.add(boxWireframe);

  return group;
}

/**
 * Create a simple conv block with inline weights
 */
function createConvBlockWithWeights(
  title: string,
  subtitle: string,
  convLayer: ConvLayer,
  color: string
): THREE.Group {
  const group = new THREE.Group();

  // Title at top
  const titleLabel = createLabel(title, new THREE.Vector3(0, 1.5, 0), 0.35, color);
  group.add(titleLabel);

  const subtitleLabel = createLabel(subtitle, new THREE.Vector3(0, 1.1, 0), 0.25, '#888888');
  group.add(subtitleLabel);

  // Compact kernel grid in center
  const kernelGrid = createCompactKernelGrid(convLayer, 0.5);
  kernelGrid.position.set(0, 0, 0);
  group.add(kernelGrid);

  // Shape info at bottom
  const shapeLabel = createLabel(
    `[${convLayer.outChannels},${convLayer.kernelSize}×${convLayer.kernelSize}]`,
    new THREE.Vector3(0, -0.9, 0),
    0.25,
    '#888888'
  );
  group.add(shapeLabel);

  // Bounding box
  const boxGeometry = new THREE.BoxGeometry(2.5, 2.5, 0.8);
  const boxEdges = new THREE.EdgesGeometry(boxGeometry);
  const boxMaterial = new THREE.LineBasicMaterial({
    color: color,
    linewidth: 2,
    transparent: true,
    opacity: 0.5,
  });
  const boxWireframe = new THREE.LineSegments(boxEdges, boxMaterial);
  boxWireframe.position.y = 0.2;
  group.add(boxWireframe);

  return group;
}

/**
 * Create a compact kernel grid showing actual weights
 * Shows a subset of kernels (max 3x3 grid of kernels)
 */
function createCompactKernelGrid(
  convLayer: ConvLayer,
  scale: number = 0.5
): THREE.Group {
  const group = new THREE.Group();

  // Display up to 3x3 kernels
  const maxDisplay = 3;
  const displayOut = Math.min(convLayer.outChannels, maxDisplay);
  const displayIn = Math.min(convLayer.inChannels, maxDisplay);

  const kernelSize = convLayer.kernelSize;
  const cellSize = scale * 0.15;
  const kernelSpacing = cellSize * (kernelSize + 1);

  for (let oc = 0; oc < displayOut; oc++) {
    for (let ic = 0; ic < displayIn; ic++) {
      const kernel = convLayer.weights[oc][ic];

      // Draw the individual kernel weights
      for (let kh = 0; kh < kernelSize; kh++) {
        for (let kw = 0; kw < kernelSize; kw++) {
          const weight = kernel[kh][kw];

          // Color based on weight value
          const absWeight = Math.abs(weight);
          const normalizedWeight = Math.tanh(absWeight * 2);

          let color: THREE.Color;
          if (weight > 0.01) {
            color = new THREE.Color(0x00ff00).multiplyScalar(0.3 + 0.7 * normalizedWeight);
          } else if (weight < -0.01) {
            color = new THREE.Color(0xff0000).multiplyScalar(0.3 + 0.7 * normalizedWeight);
          } else {
            color = new THREE.Color(0x444444);
          }

          const geometry = new THREE.BoxGeometry(cellSize, cellSize, cellSize * 0.3);
          const material = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.85,
          });
          const cube = new THREE.Mesh(geometry, material);

          // Position within the kernel
          const localX = kw * cellSize * 1.1 - (kernelSize * cellSize * 1.1) / 2;
          const localY = -kh * cellSize * 1.1 + (kernelSize * cellSize * 1.1) / 2;

          // Position within the grid of kernels
          const gridX = ic * kernelSpacing - (displayIn * kernelSpacing) / 2;
          const gridY = -oc * kernelSpacing + (displayOut * kernelSpacing) / 2;

          cube.position.set(
            gridX + localX + kernelSpacing / 2,
            gridY + localY - kernelSpacing / 2,
            0
          );

          group.add(cube);
        }
      }
    }
  }

  // Add ellipsis if there are more kernels than displayed
  if (convLayer.outChannels > maxDisplay || convLayer.inChannels > maxDisplay) {
    const ellipsisLabel = createLabel(
      '...',
      new THREE.Vector3(0, -(displayOut * kernelSpacing) / 2 - 0.3, 0),
      0.2,
      '#666666'
    );
    group.add(ellipsisLabel);
  }

  return group;
}

/**
 * Create an operation block (for downsample/upsample)
 */
function createOperationBlock(text: string, color: string): THREE.Group {
  const group = new THREE.Group();

  // Diamond shape for operations
  const shape = new THREE.Shape();
  const s = 0.5;
  shape.moveTo(0, s);
  shape.lineTo(s, 0);
  shape.lineTo(0, -s);
  shape.lineTo(-s, 0);
  shape.lineTo(0, s);

  const geometry = new THREE.ShapeGeometry(shape);
  const material = new THREE.MeshBasicMaterial({
    color: color,
    transparent: true,
    opacity: 0.7
  });
  const mesh = new THREE.Mesh(geometry, material);
  group.add(mesh);

  // Outline
  const outlineGeometry = new THREE.EdgesGeometry(geometry);
  const outlineMaterial = new THREE.LineBasicMaterial({
    color: color,
    linewidth: 2
  });
  const outline = new THREE.LineSegments(outlineGeometry, outlineMaterial);
  group.add(outline);

  // Label
  const label = createLabel(text, new THREE.Vector3(0, -0.8, 0), 0.25, color);
  group.add(label);

  return group;
}

/**
 * Create a text label with consistent styling
 */
function createLabel(
  text: string,
  position: THREE.Vector3,
  size: number,
  color: string = '#00ffff'
): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 256;
  const context = canvas.getContext('2d')!;

  context.fillStyle = 'transparent';
  context.fillRect(0, 0, canvas.width, canvas.height);

  // Better text rendering
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
    depthTest: false
  });
  const sprite = new THREE.Sprite(material);

  sprite.scale.set(size * 8, size * 2, 1);
  sprite.position.copy(position);

  return sprite;
}

/**
 * Create a flow arrow between components
 */
function createFlowArrow(
  parent: THREE.Group,
  fromPos: THREE.Vector3,
  toPos: THREE.Vector3,
  color: number
): void {
  const direction = new THREE.Vector3().subVectors(toPos, fromPos).normalize();

  // Line
  const lineGeometry = new THREE.BufferGeometry().setFromPoints([fromPos, toPos]);
  const lineMaterial = new THREE.LineBasicMaterial({
    color: color,
    linewidth: 2,
    transparent: true,
    opacity: 0.6
  });
  const line = new THREE.Line(lineGeometry, lineMaterial);
  parent.add(line);

  // Arrowhead
  const arrowGeometry = new THREE.ConeGeometry(0.15, 0.4, 8);
  const arrowMaterial = new THREE.MeshBasicMaterial({
    color: color,
    transparent: true,
    opacity: 0.8
  });
  const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
  arrow.position.copy(toPos);
  arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  parent.add(arrow);
}

/**
 * Create a small flow arrow (for within blocks)
 */
function createSmallFlowArrow(
  parent: THREE.Group,
  fromPos: THREE.Vector3,
  toPos: THREE.Vector3,
  color: number
): void {
  const direction = new THREE.Vector3().subVectors(toPos, fromPos).normalize();

  // Line
  const lineGeometry = new THREE.BufferGeometry().setFromPoints([fromPos, toPos]);
  const lineMaterial = new THREE.LineBasicMaterial({
    color: color,
    linewidth: 1,
    transparent: true,
    opacity: 0.5
  });
  const line = new THREE.Line(lineGeometry, lineMaterial);
  parent.add(line);

  // Small arrowhead
  const arrowGeometry = new THREE.ConeGeometry(0.08, 0.2, 6);
  const arrowMaterial = new THREE.MeshBasicMaterial({
    color: color,
    transparent: true,
    opacity: 0.7
  });
  const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
  arrow.position.copy(toPos);
  arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  parent.add(arrow);
}