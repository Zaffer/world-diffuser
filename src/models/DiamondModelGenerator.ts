/**
 * Generates DIAMOND model components with random initialization
 * This is for visualization purposes - showing the architecture mechanistically
 */

import {
  ConvLayer,
  GroupNormLayer,
  ActivationLayer,
  LayerType,
  ResidualBlock,
  DownsampleLayer,
  UpsampleLayer,
  UNetStage,
  MinimalUNet,
  SkipConnection
} from '../types/diamond';

/**
 * Initialize a random convolutional layer
 * Uses Xavier/Glorot initialization for weights
 */
export function createRandomConvLayer(
  inChannels: number,
  outChannels: number,
  kernelSize: number = 3,
  stride: number = 1,
  padding: number = 1
): ConvLayer {
  // Xavier initialization: weights from normal distribution with std = sqrt(2 / (fan_in + fan_out))
  const fanIn = inChannels * kernelSize * kernelSize;
  const fanOut = outChannels * kernelSize * kernelSize;
  const std = Math.sqrt(2.0 / (fanIn + fanOut));

  // Initialize weights: [outChannels][inChannels][kernelHeight][kernelWidth]
  const weights: number[][][][] = [];
  for (let oc = 0; oc < outChannels; oc++) {
    const inChannelWeights: number[][][] = [];
    for (let ic = 0; ic < inChannels; ic++) {
      const kernelWeights: number[][] = [];
      for (let kh = 0; kh < kernelSize; kh++) {
        const row: number[] = [];
        for (let kw = 0; kw < kernelSize; kw++) {
          // Box-Muller transform for normal distribution
          const u1 = Math.random();
          const u2 = Math.random();
          const normal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
          row.push(normal * std);
        }
        kernelWeights.push(row);
      }
      inChannelWeights.push(kernelWeights);
    }
    weights.push(inChannelWeights);
  }

  // Initialize biases to zero
  const bias: number[] = new Array(outChannels).fill(0);

  return {
    type: LayerType.CONV2D,
    inChannels,
    outChannels,
    kernelSize,
    stride,
    padding,
    weights,
    bias,
  };
}

/**
 * Initialize a random group normalization layer
 */
export function createRandomGroupNormLayer(
  numChannels: number,
  numGroups: number = 8,
  epsilon: number = 1e-5
): GroupNormLayer {
  // Initialize gamma (scale) to 1 and beta (shift) to 0
  const gamma: number[] = new Array(numChannels).fill(1.0);
  const beta: number[] = new Array(numChannels).fill(0.0);

  return {
    type: LayerType.GROUP_NORM,
    numGroups,
    numChannels,
    gamma,
    beta,
    epsilon,
  };
}

/**
 * Create an activation layer
 */
export function createActivationLayer(activation: 'relu' | 'silu' | 'gelu' | 'tanh' = 'silu'): ActivationLayer {
  return {
    type: LayerType.ACTIVATION,
    activation,
  };
}

/**
 * Create a complete residual block (the fundamental building block of U-Net)
 */
export function createResidualBlock(
  inChannels: number,
  outChannels: number,
  kernelSize: number = 3
): ResidualBlock {
  return {
    conv1: createRandomConvLayer(inChannels, outChannels, kernelSize),
    norm1: createRandomGroupNormLayer(outChannels),
    activation1: createActivationLayer('silu'),
    conv2: createRandomConvLayer(outChannels, outChannels, kernelSize),
    norm2: createRandomGroupNormLayer(outChannels),
    activation2: createActivationLayer('silu'),
  };
}

/**
 * Create a downsampling layer using strided convolution
 */
export function createDownsampleLayer(
  channels: number,
  factor: number = 2
): DownsampleLayer {
  return {
    type: LayerType.DOWNSAMPLE,
    method: 'conv_stride',
    factor,
    conv: createRandomConvLayer(channels, channels, 3, factor, 1),
  };
}

/**
 * Create an upsampling layer using transposed convolution
 */
export function createUpsampleLayer(
  inChannels: number,
  outChannels: number,
  factor: number = 2
): UpsampleLayer {
  return {
    type: LayerType.UPSAMPLE,
    method: 'conv_transpose',
    factor,
    conv: createRandomConvLayer(inChannels, outChannels, factor, factor, 0),
  };
}

/**
 * Create a minimal U-Net architecture for DIAMOND
 * Configuration: 4x4 input, [8, 16] channels, 1 encoder/decoder stage
 * Total parameters: ~1,200 (all visible!)
 */
export function createMinimalUNet(): MinimalUNet {
  // Input: 4x4 RGB (3 channels)
  const inputSpatialSize = 4;

  // Channel progression: 3 → 8 → 16 → 8 → 3
  const encoderChannels = 8;
  const bottleneckChannels = 16;

  // Input convolution: 3 → 8 channels
  const inputConv = createRandomConvLayer(3, encoderChannels, 3, 1, 1);

  // Encoder stage: 8 → 8 channels, 4x4 spatial
  const encoderBlock = createResidualBlock(encoderChannels, encoderChannels, 3);
  const encoderDownsample = createDownsampleLayer(encoderChannels, 2);

  const encoder: UNetStage = {
    stageName: 'Encoder_Stage0',
    blocks: [encoderBlock],
    downsample: encoderDownsample,
    spatialShape: [inputSpatialSize, inputSpatialSize],
    channelCount: encoderChannels,
  };

  // Bottleneck: 8 → 16 channels, 2x2 spatial
  const bottleneck = createResidualBlock(encoderChannels, bottleneckChannels, 3);

  // Decoder stage: upsample then 16 → 8 channels
  const decoderUpsample = createUpsampleLayer(bottleneckChannels, encoderChannels, 2);
  // After skip connection concatenation: 8 (from upsample) + 8 (from skip) = 16 channels input
  const decoderBlock = createResidualBlock(encoderChannels * 2, encoderChannels, 3);

  const decoder: UNetStage = {
    stageName: 'Decoder_Stage0',
    blocks: [decoderBlock],
    upsample: decoderUpsample,
    spatialShape: [inputSpatialSize, inputSpatialSize],
    channelCount: encoderChannels,
  };

  // Output convolution: 8 → 3 channels
  const outputConv = createRandomConvLayer(encoderChannels, 3, 3, 1, 1);

  // Skip connection from encoder to decoder
  const skipConnections: SkipConnection[] = [
    {
      id: 'Skip_E0_D0',
      fromBlockName: 'Encoder_Block0',
      toBlockName: 'Decoder_Block0',
      fromShape: [encoderChannels, inputSpatialSize, inputSpatialSize],
      toShape: [encoderChannels, inputSpatialSize, inputSpatialSize],
    },
  ];

  return {
    inputConv,
    encoder,
    bottleneck,
    decoder,
    skipConnections,
    outputConv,
  };
}

/**
 * Create a minimal DIAMOND model structure (legacy function - kept for compatibility)
 * Start with just one residual block for now
 */
export function createMinimalDiamondModel() {
  // For Atari frames: typically 64x64 RGB images
  // Start with a single residual block: 3 channels (RGB) -> 64 channels
  const firstBlock = createResidualBlock(3, 64, 3);

  return {
    blocks: [firstBlock],
    inputShape: { channels: 3, height: 64, width: 64 },
  };
}
