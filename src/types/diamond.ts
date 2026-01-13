/**
 * Types for DIAMOND diffusion world model architecture visualization
 */

/**
 * Layer types in the U-Net architecture
 */
export enum LayerType {
  CONV2D = 'conv2d',
  GROUP_NORM = 'group_norm',
  ACTIVATION = 'activation',
  RESIDUAL_CONNECTION = 'residual_connection',
  ATTENTION = 'attention',
  DOWNSAMPLE = 'downsample',
  UPSAMPLE = 'upsample',
}

/**
 * Represents a convolutional layer
 */
export interface ConvLayer {
  type: LayerType.CONV2D;
  inChannels: number;
  outChannels: number;
  kernelSize: number; // e.g., 3 for 3x3
  stride: number;
  padding: number;
  weights: number[][][][]; // [outChannels][inChannels][kernelHeight][kernelWidth]
  bias: number[]; // [outChannels]
}

/**
 * Represents a group normalization layer
 */
export interface GroupNormLayer {
  type: LayerType.GROUP_NORM;
  numGroups: number;
  numChannels: number;
  gamma: number[]; // [numChannels] - scale parameter
  beta: number[]; // [numChannels] - shift parameter
  epsilon: number;
}

/**
 * Represents an activation function layer
 */
export interface ActivationLayer {
  type: LayerType.ACTIVATION;
  activation: 'relu' | 'silu' | 'gelu' | 'tanh';
}

/**
 * Represents a residual connection
 */
export interface ResidualConnection {
  type: LayerType.RESIDUAL_CONNECTION;
  fromLayerIndex: number;
  toLayerIndex: number;
}

/**
 * Represents a feature map (activations at any layer)
 */
export interface FeatureMap {
  channels: number;
  height: number;
  width: number;
  data: number[][][]; // [channels][height][width]
}

/**
 * Represents a single residual block in the U-Net
 */
export interface ResidualBlock {
  conv1: ConvLayer;
  norm1: GroupNormLayer;
  activation1: ActivationLayer;
  conv2: ConvLayer;
  norm2: GroupNormLayer;
  activation2: ActivationLayer;
  residualConnection?: ResidualConnection;
}

/**
 * Configuration for visualizing a layer
 */
export interface LayerVisualizationConfig {
  showWeights: boolean;
  showBiases: boolean;
  showActivations: boolean;
  colorScheme: 'redgreen' | 'blueorange' | 'viridis';
  weightScale: number;
}

/**
 * Default visualization configuration
 */
export const DEFAULT_LAYER_VIS_CONFIG: LayerVisualizationConfig = {
  showWeights: true,
  showBiases: true,
  showActivations: true,
  colorScheme: 'redgreen',
  weightScale: 1.0,
};

/**
 * Represents a downsampling layer (reduces spatial dimensions)
 */
export interface DownsampleLayer {
  type: LayerType.DOWNSAMPLE;
  method: 'conv_stride' | 'maxpool' | 'avgpool';
  factor: number; // 2 for 2x downsampling
  conv?: ConvLayer; // if method is conv_stride
}

/**
 * Represents an upsampling layer (increases spatial dimensions)
 */
export interface UpsampleLayer {
  type: LayerType.UPSAMPLE;
  method: 'conv_transpose' | 'nearest' | 'bilinear';
  factor: number; // 2 for 2x upsampling
  conv?: ConvLayer; // if method is conv_transpose
}

/**
 * Represents a skip connection between encoder and decoder
 */
export interface SkipConnection {
  id: string;
  fromBlockName: string;
  toBlockName: string;
  fromShape: [number, number, number]; // [C, H, W]
  toShape: [number, number, number];
}

/**
 * Represents a stage in the U-Net (encoder or decoder)
 */
export interface UNetStage {
  stageName: string;
  blocks: ResidualBlock[];
  downsample?: DownsampleLayer;
  upsample?: UpsampleLayer;
  spatialShape: [number, number]; // [H, W]
  channelCount: number;
}

/**
 * Complete minimal U-Net architecture
 */
export interface MinimalUNet {
  inputConv: ConvLayer;
  encoder: UNetStage;
  bottleneck: ResidualBlock;
  decoder: UNetStage;
  skipConnections: SkipConnection[];
  outputConv: ConvLayer;
}

/**
 * Configuration for visualizing the full architecture
 */
export interface ArchitectureVisConfig extends LayerVisualizationConfig {
  showBlockBounds: boolean;
  showSkipConnections: boolean;
  showTensorShapes: boolean;
  showParameterCounts: boolean;
  blockSpacing: number; // Y-axis spacing between blocks
  kernelDisplayLimit: number; // Max kernels to show per layer
}

/**
 * Default architecture visualization configuration
 */
export const DEFAULT_ARCH_VIS_CONFIG: ArchitectureVisConfig = {
  ...DEFAULT_LAYER_VIS_CONFIG,
  showBlockBounds: true,
  showSkipConnections: true,
  showTensorShapes: true,
  showParameterCounts: false,
  blockSpacing: 2.0,
  kernelDisplayLimit: 8,
};
