/**
 * Types for the ultra-minimal diffusion model
 * Designed for complete weight visibility (~100-200 weights)
 */

/**
 * A single 3x3 convolution kernel
 */
export interface Kernel3x3 {
  weights: number[][]; // [3][3]
}

/**
 * Conv layer with explicit weight storage
 * weights[outChannel][inChannel] = Kernel3x3
 */
export interface TinyConvLayer {
  name: string;
  inChannels: number;
  outChannels: number;
  kernels: Kernel3x3[][]; // [outChannels][inChannels]
  bias: number[];         // [outChannels]
}

/**
 * Time embedding layer (simple linear projection)
 */
export interface TimeEmbedding {
  weights: number[];  // [outputDim] - single input, multiple outputs
  bias: number[];     // [outputDim]
  outputDim: number;
}

/**
 * Activation tensor (feature map) at any point in the network
 */
export interface ActivationTensor {
  data: number[][][]; // [channels][height][width]
  channels: number;
  height: number;
  width: number;
}

/**
 * Complete tiny U-Net model
 * 
 * Architecture:
 * - Input: 2×2 grayscale (1 channel)
 * - Channels: 1 → 2 → 2 → 2 → 1
 * - Single encoder/decoder stage
 * - Skip connection from encoder to decoder
 * 
 * Total weights breakdown:
 * - timeEmbed: 2 weights + 2 bias = 4
 * - inputConv: 1×2×9 + 2 bias = 20
 * - encoderConv: 2×2×9 + 2 bias = 38
 * - bottleneckConv: 2×2×9 + 2 bias = 38
 * - decoderConv: 4×2×9 + 2 bias = 74 (4 in due to skip concat)
 * - outputConv: 2×1×9 + 1 bias = 19
 * 
 * Total: ~193 weights
 */
export interface TinyUNet {
  // Time embedding
  timeEmbed: TimeEmbedding;
  
  // Main path
  inputConv: TinyConvLayer;    // 1ch → 2ch
  encoderConv: TinyConvLayer;  // 2ch → 2ch
  bottleneckConv: TinyConvLayer; // 2ch → 2ch (at 1×1 spatial)
  decoderConv: TinyConvLayer;  // 4ch → 2ch (after skip concat)
  outputConv: TinyConvLayer;   // 2ch → 1ch
}

/**
 * Stores all activations during a forward pass
 */
export interface ForwardPassState {
  // Input
  noisyInput: ActivationTensor;     // 2×2×1
  timestep: number;                  // scalar t ∈ [0, 1]
  
  // After each layer
  afterInputConv: ActivationTensor;  // 2×2×2
  afterEncoder: ActivationTensor;    // 2×2×2 (saved for skip)
  afterDownsample: ActivationTensor; // 1×1×2
  afterBottleneck: ActivationTensor; // 1×1×2
  afterUpsample: ActivationTensor;   // 2×2×2
  afterSkipConcat: ActivationTensor; // 2×2×4
  afterDecoder: ActivationTensor;    // 2×2×2
  output: ActivationTensor;          // 2×2×1 (predicted noise)
}

/**
 * Configuration for tiny model visualization
 */
export interface TinyVisConfig {
  // Layout
  layerSpacing: number;      // Horizontal space between layers
  channelSpacing: number;    // Z-space between channels
  kernelScale: number;       // Size of kernel weight cubes
  activationScale: number;   // Size of activation planes
  
  // Colors
  positiveColor: number;     // Color for positive weights
  negativeColor: number;     // Color for negative weights
  activationColormap: 'viridis' | 'plasma' | 'grayscale';
  
  // Display options
  showWeightValues: boolean;
  showActivations: boolean;
  showDataFlow: boolean;
  showTimeEmbedding: boolean;
  showSkipConnection: boolean;
}

export const DEFAULT_TINY_VIS_CONFIG: TinyVisConfig = {
  layerSpacing: 6,
  channelSpacing: 1.5,
  kernelScale: 0.35,
  activationScale: 1.0,
  
  positiveColor: 0x22ccff,  // More saturated cyan for positive
  negativeColor: 0xff5500,  // Deeper orange for negative
  activationColormap: 'viridis',
  
  showWeightValues: true,
  showActivations: true,
  showDataFlow: true,
  showTimeEmbedding: true,
  showSkipConnection: true,
};
