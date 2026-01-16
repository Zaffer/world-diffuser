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
  projection?: BlockProjection; // Optional per-block time conditioning
}

/**
 * Linear layer for MLP
 */
export interface LinearLayer {
  weights: number[][]; // [outputDim][inputDim]
  bias: number[];      // [outputDim]
  inputDim: number;
  outputDim: number;
}

/**
 * Time embedding MLP following EDM/DIAMOND architecture
 * Uses cnoise(σ) = 0.25 * ln(σ) as input (fixed formula, not learned)
 * Then passes through a 2-layer MLP to create shared time embedding
 */
export interface TimeEmbeddingMLP {
  hiddenLayer: LinearLayer;   // cnoise → hidden (e.g., 1 → 4)
  outputLayer: LinearLayer;   // hidden → embedding (e.g., 4 → 8)
}

/**
 * Per-block projection layer
 * Projects shared time embedding to block-specific conditioning
 */
export interface BlockProjection {
  weights: number[][]; // [outputDim][embeddingDim]
  bias: number[];      // [outputDim]
  embeddingDim: number;
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
 * Complete tiny U-Net model with EDM-style time conditioning
 *
 * Architecture:
 * - Input: 2×2 grayscale (1 channel)
 * - Channels: 1 → 2 → 2 → 2 → 1
 * - Single encoder/decoder stage
 * - Skip connection from encoder to decoder
 * - EDM time conditioning: σ → cnoise(σ) → MLP → per-block projections
 *
 * Total weights breakdown (approximate):
 * - timeEmbedMLP:
 *   - hidden: 1×4 + 4 bias = 8
 *   - output: 4×8 + 8 bias = 40
 * - Per-block projections (4 blocks × 8→2): 4 × (8×2 + 2) = 72
 * - inputConv: 1×2×9 + 2 bias = 20
 * - encoderConv: 2×2×9 + 2 bias = 38
 * - bottleneckConv: 2×2×9 + 2 bias = 38
 * - decoderConv: 4×2×9 + 2 bias = 74 (4 in due to skip concat)
 * - outputConv: 2×1×9 + 1 bias = 19
 *
 * Total: ~309 weights
 */
export interface TinyUNet {
  // Time embedding MLP (shared across all blocks)
  timeEmbedMLP: TimeEmbeddingMLP;

  // Main path (now with per-block projections)
  inputConv: TinyConvLayer;    // 1ch → 2ch
  encoderConv: TinyConvLayer;  // 2ch → 2ch
  bottleneckConv: TinyConvLayer; // 2ch → 2ch (at 1×1 spatial)
  decoderConv: TinyConvLayer;  // 4ch → 2ch (after skip concat)
  outputConv: TinyConvLayer;   // 2ch → 1ch (no time conditioning)
}

/**
 * Stores all activations during a forward pass
 */
export interface ForwardPassState {
  // Input
  noisyInput: ActivationTensor;     // 2×2×1
  timestep: number;                  // scalar σ (noise level) ∈ [0, 1]

  // Time conditioning (for visualization)
  cnoise: number;                    // Computed cnoise(σ) = 0.25 * ln(σ)
  mlpHidden: number[];               // Hidden layer activations
  sharedEmbedding: number[];         // Output of shared MLP

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
  negativeColor: 0xff5522,  // Deeper orange for negative
  activationColormap: 'viridis',
  
  showWeightValues: true,
  showActivations: true,
  showDataFlow: true,
  showTimeEmbedding: true,
  showSkipConnection: true,
};
