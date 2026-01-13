/**
 * Full U-Net architecture visualization for DIAMOND
 * Redesigned with clear horizontal U-shape layout
 */

import * as THREE from 'three';
import {
  MinimalUNet,
  ArchitectureVisConfig,
  DEFAULT_ARCH_VIS_CONFIG,
} from '../types/diamond';

/**
 * Create the complete minimal U-Net architecture visualization
 * Horizontal U-shape layout: Encoder (left) → Bottleneck (bottom) → Decoder (right)
 */
export function createMinimalUNetVisualization(
  _model: MinimalUNet,
  config: ArchitectureVisConfig = DEFAULT_ARCH_VIS_CONFIG
): THREE.Group {
  const group = new THREE.Group();

  // Base unit for all spacing
  const UNIT = 3;

  // ========== LAYOUT POSITIONS ==========
  // Horizontal U-shape: Left (encoder) → Bottom (bottleneck) → Right (decoder)

  const positions = {
    input: new THREE.Vector3(-UNIT * 3, UNIT * 3, 0),
    encoder: new THREE.Vector3(-UNIT * 3, 0, 0),
    downsample: new THREE.Vector3(-UNIT * 3, -UNIT * 1.5, 0),
    bottleneck: new THREE.Vector3(0, -UNIT * 3, 0),
    upsample: new THREE.Vector3(UNIT * 3, -UNIT * 1.5, 0),
    decoder: new THREE.Vector3(UNIT * 3, 0, 0),
    output: new THREE.Vector3(UNIT * 3, UNIT * 3, 0),
  };

  // ========== INPUT CONVOLUTION ==========
  const inputBlock = createAbstractBlock(
    'Input Conv',
    '3→8 ch',
    '[8,4,4]',
    '#4488ff',
    new THREE.Vector3(1.2, 1.2, 0.8)
  );
  inputBlock.position.copy(positions.input);
  group.add(inputBlock);

  // ========== ENCODER STAGE ==========
  const encoderBlock = createAbstractBlock(
    'Encoder',
    'Residual Block',
    '[8,4,4]',
    '#00aaff',
    new THREE.Vector3(1.8, 1.8, 1.2)
  );
  encoderBlock.position.copy(positions.encoder);
  group.add(encoderBlock);

  // Downsample indicator
  const downsampleBlock = createOperationBlock('↓ Downsample ×2', '#ff9900');
  downsampleBlock.position.copy(positions.downsample);
  group.add(downsampleBlock);

  // ========== BOTTLENECK STAGE ==========
  const bottleneckBlock = createAbstractBlock(
    'Bottleneck',
    'Residual Block',
    '[16,2,2]',
    '#aa00ff',
    new THREE.Vector3(2.2, 2.2, 1.4)
  );
  bottleneckBlock.position.copy(positions.bottleneck);
  group.add(bottleneckBlock);

  // Upsample indicator
  const upsampleBlock = createOperationBlock('↑ Upsample ×2', '#00ff99');
  upsampleBlock.position.copy(positions.upsample);
  group.add(upsampleBlock);

  // ========== DECODER STAGE ==========
  const decoderBlock = createAbstractBlock(
    'Decoder',
    'Residual Block',
    '[8,4,4]',
    '#00ff88',
    new THREE.Vector3(1.8, 1.8, 1.2)
  );
  decoderBlock.position.copy(positions.decoder);
  group.add(decoderBlock);

  // ========== OUTPUT CONVOLUTION ==========
  const outputBlock = createAbstractBlock(
    'Output Conv',
    '8→3 ch',
    '[3,4,4]',
    '#44ff88',
    new THREE.Vector3(1.2, 1.2, 0.8)
  );
  outputBlock.position.copy(positions.output);
  group.add(outputBlock);

  // ========== SKIP CONNECTION ==========
  if (config.showSkipConnections) {
    const skipCurve = new THREE.QuadraticBezierCurve3(
      positions.encoder.clone().add(new THREE.Vector3(1.2, 0, 0)),
      new THREE.Vector3(0, UNIT * 1.5, 0),
      positions.decoder.clone().add(new THREE.Vector3(-1.2, 0, 0))
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
      0.4,
      '#ff00ff'
    );
    group.add(skipLabel);
  }

  // ========== FLOW ARROWS ==========
  const arrowColor = 0xffffff;

  // Input → Encoder
  createFlowArrow(
    group,
    positions.input.clone().add(new THREE.Vector3(0, -0.8, 0)),
    positions.encoder.clone().add(new THREE.Vector3(0, 1.2, 0)),
    arrowColor
  );

  // Encoder → Downsample
  createFlowArrow(
    group,
    positions.encoder.clone().add(new THREE.Vector3(0, -1.2, 0)),
    positions.downsample.clone().add(new THREE.Vector3(0, 0.3, 0)),
    arrowColor
  );

  // Downsample → Bottleneck
  createFlowArrow(
    group,
    positions.downsample.clone().add(new THREE.Vector3(0.5, -0.3, 0)),
    positions.bottleneck.clone().add(new THREE.Vector3(-1.5, 0.5, 0)),
    arrowColor
  );

  // Bottleneck → Upsample
  createFlowArrow(
    group,
    positions.bottleneck.clone().add(new THREE.Vector3(1.5, 0.5, 0)),
    positions.upsample.clone().add(new THREE.Vector3(-0.5, -0.3, 0)),
    arrowColor
  );

  // Upsample → Decoder
  createFlowArrow(
    group,
    positions.upsample.clone().add(new THREE.Vector3(0, 0.3, 0)),
    positions.decoder.clone().add(new THREE.Vector3(0, -1.2, 0)),
    arrowColor
  );

  // Decoder → Output
  createFlowArrow(
    group,
    positions.decoder.clone().add(new THREE.Vector3(0, 1.2, 0)),
    positions.output.clone().add(new THREE.Vector3(0, -0.8, 0)),
    arrowColor
  );

  // ========== STAGE LABELS ==========
  const encoderStageLabel = createLabel(
    'ENCODER PATH',
    new THREE.Vector3(-UNIT * 3, UNIT * 4.5, 0),
    0.5,
    '#00aaff'
  );
  group.add(encoderStageLabel);

  const decoderStageLabel = createLabel(
    'DECODER PATH',
    new THREE.Vector3(UNIT * 3, UNIT * 4.5, 0),
    0.5,
    '#00ff88'
  );
  group.add(decoderStageLabel);

  return group;
}

/**
 * Create an abstract block representation (clean box with labels)
 */
function createAbstractBlock(
  title: string,
  subtitle: string,
  shape: string,
  color: string,
  size: THREE.Vector3
): THREE.Group {
  const group = new THREE.Group();

  // Main box with wireframe
  const geometry = new THREE.BoxGeometry(size.x, size.y, size.z);

  // Solid fill with transparency - INTERACTABLE
  const fillMaterial = new THREE.MeshBasicMaterial({
    color: color,
    transparent: true,
    opacity: 0.2,
  });
  const fillMesh = new THREE.Mesh(geometry, fillMaterial);

  // Make this mesh interactable
  fillMesh.userData = {
    type: 'diamond_block', // Will match InteractableType.DIAMOND_BLOCK
    blockName: title,
    blockSubtitle: subtitle,
    blockShape: shape,
    blockColor: color,
  };

  group.add(fillMesh);

  // Wireframe edges
  const edges = new THREE.EdgesGeometry(geometry);
  const lineMaterial = new THREE.LineBasicMaterial({
    color: color,
    linewidth: 2
  });
  const wireframe = new THREE.LineSegments(edges, lineMaterial);
  group.add(wireframe);

  // Title label (top)
  const titleLabel = createLabel(
    title,
    new THREE.Vector3(0, size.y / 2 + 0.5, 0),
    0.35,
    color
  );
  group.add(titleLabel);

  // Subtitle label (middle)
  const subtitleLabel = createLabel(
    subtitle,
    new THREE.Vector3(0, 0, size.z / 2 + 0.3),
    0.25,
    '#cccccc'
  );
  group.add(subtitleLabel);

  // Shape label (bottom)
  const shapeLabel = createLabel(
    shape,
    new THREE.Vector3(0, -size.y / 2 - 0.5, 0),
    0.3,
    '#888888'
  );
  group.add(shapeLabel);

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
    depthTest: false // Always render on top
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
