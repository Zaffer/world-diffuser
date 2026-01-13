/**
 * Full U-Net architecture visualization for DIAMOND
 * Lays out all blocks vertically with proper spacing
 */

import * as THREE from 'three';
import {
  MinimalUNet,
  ArchitectureVisConfig,
  DEFAULT_ARCH_VIS_CONFIG,
} from '../types/diamond';
import {
  createResidualBlockVis,
  createDownsampleVis,
  createUpsampleVis,
  createSkipConnectionVis,
  createConvLayerVisualization,
} from './DiamondVis';

/**
 * Create the complete minimal U-Net architecture visualization
 * Vertical layout: InputConv → Encoder → Bottleneck → Decoder → OutputConv
 */
export function createMinimalUNetVisualization(
  model: MinimalUNet,
  config: ArchitectureVisConfig = DEFAULT_ARCH_VIS_CONFIG
): THREE.Group {
  const group = new THREE.Group();

  // Coordinate system (from plan):
  // Y-axis: vertical stacking
  // +6: OutputConv
  // +4: Decoder_Block0
  // +2: Decoder_Upsample
  //  0: Bottleneck (origin)
  // -2: Encoder_Downsample
  // -4: Encoder_Block0
  // -6: InputConv

  const blockSpacing = config.blockSpacing;

  // Store positions for skip connections
  const blockPositions: Record<string, THREE.Vector3> = {};

  // InputConv at Y = -6
  const inputConvVis = createConvLayerVisualization(model.inputConv, config);
  inputConvVis.position.set(0, -6, 0);
  inputConvVis.scale.setScalar(0.4);
  group.add(inputConvVis);

  // Add label for InputConv
  const inputLabel = createStageLabel('InputConv: 3→8 ch', new THREE.Vector3(0, -5.5, 0));
  group.add(inputLabel);

  // Encoder Block at Y = -4
  const encoderBlock = model.encoder.blocks[0];
  const encoderBlockVis = createResidualBlockVis(
    encoderBlock,
    'Encoder_Block0',
    new THREE.Vector3(0, -4, 0),
    config
  );
  group.add(encoderBlockVis);
  blockPositions['Encoder_Block0'] = new THREE.Vector3(0, -4, 0);

  // Add label showing spatial shape
  const encoderLabel = createStageLabel(
    `[8, 4, 4]`,
    new THREE.Vector3(0, -3.3, 0)
  );
  group.add(encoderLabel);

  // Encoder Downsample at Y = -2
  if (model.encoder.downsample) {
    const downsampleVis = createDownsampleVis(
      model.encoder.downsample,
      new THREE.Vector3(0, -2, 0)
    );
    group.add(downsampleVis);
  }

  // Bottleneck at Y = 0 (center)
  const bottleneckVis = createResidualBlockVis(
    model.bottleneck,
    'Bottleneck',
    new THREE.Vector3(0, 0, 0),
    config
  );
  group.add(bottleneckVis);

  // Add label showing bottleneck shape
  const bottleneckLabel = createStageLabel(
    `[16, 2, 2]`,
    new THREE.Vector3(0, 0.7, 0)
  );
  group.add(bottleneckLabel);

  // Decoder Upsample at Y = +2
  if (model.decoder.upsample) {
    const upsampleVis = createUpsampleVis(
      model.decoder.upsample,
      new THREE.Vector3(0, 2, 0)
    );
    group.add(upsampleVis);
  }

  // Decoder Block at Y = +4
  const decoderBlock = model.decoder.blocks[0];
  const decoderBlockVis = createResidualBlockVis(
    decoderBlock,
    'Decoder_Block0',
    new THREE.Vector3(0, 4, 0),
    config
  );
  group.add(decoderBlockVis);
  blockPositions['Decoder_Block0'] = new THREE.Vector3(0, 4, 0);

  // Add label showing spatial shape
  const decoderLabel = createStageLabel(
    `[8, 4, 4]`,
    new THREE.Vector3(0, 4.7, 0)
  );
  group.add(decoderLabel);

  // OutputConv at Y = +6
  const outputConvVis = createConvLayerVisualization(model.outputConv, config);
  outputConvVis.position.set(0, 6, 0);
  outputConvVis.scale.setScalar(0.4);
  group.add(outputConvVis);

  // Add label for OutputConv
  const outputLabel = createStageLabel('OutputConv: 8→3 ch', new THREE.Vector3(0, 6.5, 0));
  group.add(outputLabel);

  // Skip connections
  if (config.showSkipConnections) {
    model.skipConnections.forEach((skip) => {
      const fromPos = blockPositions[skip.fromBlockName];
      const toPos = blockPositions[skip.toBlockName];

      if (fromPos && toPos) {
        const skipVis = createSkipConnectionVis(skip, fromPos, toPos);
        group.add(skipVis);
      }
    });
  }

  // Add stage labels on the side
  const encoderStageLabel = createStageLabel('ENCODER', new THREE.Vector3(-3, -4, 0), 0.15);
  group.add(encoderStageLabel);

  const bottleneckStageLabel = createStageLabel('BOTTLENECK', new THREE.Vector3(-3, 0, 0), 0.15);
  group.add(bottleneckStageLabel);

  const decoderStageLabel = createStageLabel('DECODER', new THREE.Vector3(-3, 4, 0), 0.15);
  group.add(decoderStageLabel);

  return group;
}

/**
 * Create a text label for stage/component names
 */
function createStageLabel(text: string, position: THREE.Vector3, size: number = 0.1): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 128;
  const context = canvas.getContext('2d')!;

  context.fillStyle = '#1a1a1a';
  context.fillRect(0, 0, canvas.width, canvas.height);

  context.font = 'bold 32px monospace';
  context.fillStyle = '#00ffff';
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
