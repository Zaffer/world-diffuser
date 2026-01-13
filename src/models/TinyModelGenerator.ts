/**
 * Tiny U-Net model generator and forward pass implementation
 * All weights are explicit and trackable
 */

import {
  TinyUNet,
  TinyConvLayer,
  TimeEmbedding,
  Kernel3x3,
  ActivationTensor,
  ForwardPassState,
} from '../types/tiny';

/**
 * Xavier/Glorot initialization for a single weight
 */
function xavierInit(fanIn: number, fanOut: number): number {
  const std = Math.sqrt(2.0 / (fanIn + fanOut));
  // Box-Muller for normal distribution
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * std;
}

/**
 * Create a 3×3 kernel with Xavier initialization
 */
function createKernel3x3(fanIn: number, fanOut: number): Kernel3x3 {
  const weights: number[][] = [];
  for (let i = 0; i < 3; i++) {
    const row: number[] = [];
    for (let j = 0; j < 3; j++) {
      row.push(xavierInit(fanIn * 9, fanOut * 9));
    }
    weights.push(row);
  }
  return { weights };
}

/**
 * Create a tiny conv layer
 */
function createTinyConvLayer(
  name: string,
  inChannels: number,
  outChannels: number
): TinyConvLayer {
  const kernels: Kernel3x3[][] = [];
  
  for (let oc = 0; oc < outChannels; oc++) {
    const inKernels: Kernel3x3[] = [];
    for (let ic = 0; ic < inChannels; ic++) {
      inKernels.push(createKernel3x3(inChannels, outChannels));
    }
    kernels.push(inKernels);
  }
  
  // Initialize bias to zero
  const bias = new Array(outChannels).fill(0);
  
  return { name, inChannels, outChannels, kernels, bias };
}

/**
 * Create time embedding layer
 */
function createTimeEmbedding(outputDim: number): TimeEmbedding {
  const weights: number[] = [];
  const bias: number[] = [];
  
  for (let i = 0; i < outputDim; i++) {
    weights.push(xavierInit(1, outputDim));
    bias.push(0);
  }
  
  return { weights, bias, outputDim };
}

/**
 * Create the complete tiny U-Net model
 */
export function createTinyUNet(): TinyUNet {
  return {
    timeEmbed: createTimeEmbedding(2),
    inputConv: createTinyConvLayer('inputConv', 1, 2),
    encoderConv: createTinyConvLayer('encoderConv', 2, 2),
    bottleneckConv: createTinyConvLayer('bottleneckConv', 2, 2),
    decoderConv: createTinyConvLayer('decoderConv', 4, 2), // 4 = 2 + 2 skip
    outputConv: createTinyConvLayer('outputConv', 2, 1),
  };
}

/**
 * Count total parameters in the model
 */
export function countParameters(model: TinyUNet): number {
  let count = 0;
  
  // Time embedding
  count += model.timeEmbed.weights.length;
  count += model.timeEmbed.bias.length;
  
  // Conv layers
  const convLayers = [
    model.inputConv,
    model.encoderConv,
    model.bottleneckConv,
    model.decoderConv,
    model.outputConv,
  ];
  
  for (const layer of convLayers) {
    // Kernels: outChannels × inChannels × 3 × 3
    count += layer.outChannels * layer.inChannels * 9;
    // Bias: outChannels
    count += layer.outChannels;
  }
  
  return count;
}

// ============ FORWARD PASS IMPLEMENTATION ============

/**
 * Create an empty activation tensor
 */
function createActivationTensor(channels: number, height: number, width: number): ActivationTensor {
  const data: number[][][] = [];
  for (let c = 0; c < channels; c++) {
    const channel: number[][] = [];
    for (let h = 0; h < height; h++) {
      channel.push(new Array(width).fill(0));
    }
    data.push(channel);
  }
  return { data, channels, height, width };
}

/**
 * Apply 3×3 convolution with padding=1 (same output size)
 */
function applyConv2d(input: ActivationTensor, layer: TinyConvLayer): ActivationTensor {
  const { height, width } = input;
  const output = createActivationTensor(layer.outChannels, height, width);
  
  for (let oc = 0; oc < layer.outChannels; oc++) {
    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        let sum = layer.bias[oc];
        
        // Sum over input channels and kernel
        for (let ic = 0; ic < layer.inChannels; ic++) {
          const kernel = layer.kernels[oc][ic].weights;
          
          for (let kh = 0; kh < 3; kh++) {
            for (let kw = 0; kw < 3; kw++) {
              const ih = h + kh - 1; // -1 for padding
              const iw = w + kw - 1;
              
              // Zero padding
              if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                sum += input.data[ic][ih][iw] * kernel[kh][kw];
              }
            }
          }
        }
        
        output.data[oc][h][w] = sum;
      }
    }
  }
  
  return output;
}

/**
 * SiLU activation (x * sigmoid(x))
 */
function applySiLU(input: ActivationTensor): ActivationTensor {
  const output = createActivationTensor(input.channels, input.height, input.width);
  
  for (let c = 0; c < input.channels; c++) {
    for (let h = 0; h < input.height; h++) {
      for (let w = 0; w < input.width; w++) {
        const x = input.data[c][h][w];
        const sigmoid = 1 / (1 + Math.exp(-x));
        output.data[c][h][w] = x * sigmoid;
      }
    }
  }
  
  return output;
}

/**
 * Downsample 2×2 → 1×1 using average pooling
 */
function downsample2x(input: ActivationTensor): ActivationTensor {
  const output = createActivationTensor(
    input.channels,
    Math.floor(input.height / 2),
    Math.floor(input.width / 2)
  );
  
  for (let c = 0; c < input.channels; c++) {
    for (let h = 0; h < output.height; h++) {
      for (let w = 0; w < output.width; w++) {
        const sum =
          input.data[c][h * 2][w * 2] +
          input.data[c][h * 2 + 1][w * 2] +
          input.data[c][h * 2][w * 2 + 1] +
          input.data[c][h * 2 + 1][w * 2 + 1];
        output.data[c][h][w] = sum / 4;
      }
    }
  }
  
  return output;
}

/**
 * Upsample 1×1 → 2×2 using nearest neighbor
 */
function upsample2x(input: ActivationTensor): ActivationTensor {
  const output = createActivationTensor(
    input.channels,
    input.height * 2,
    input.width * 2
  );
  
  for (let c = 0; c < input.channels; c++) {
    for (let h = 0; h < input.height; h++) {
      for (let w = 0; w < input.width; w++) {
        const val = input.data[c][h][w];
        output.data[c][h * 2][w * 2] = val;
        output.data[c][h * 2 + 1][w * 2] = val;
        output.data[c][h * 2][w * 2 + 1] = val;
        output.data[c][h * 2 + 1][w * 2 + 1] = val;
      }
    }
  }
  
  return output;
}

/**
 * Concatenate two tensors along channel dimension
 */
function concat(a: ActivationTensor, b: ActivationTensor): ActivationTensor {
  const output = createActivationTensor(
    a.channels + b.channels,
    a.height,
    a.width
  );
  
  for (let c = 0; c < a.channels; c++) {
    output.data[c] = a.data[c].map(row => [...row]);
  }
  for (let c = 0; c < b.channels; c++) {
    output.data[a.channels + c] = b.data[c].map(row => [...row]);
  }
  
  return output;
}

/**
 * Add time embedding to activations (broadcast across spatial dims)
 */
function addTimeEmbedding(
  input: ActivationTensor,
  timeEmbed: TimeEmbedding,
  timestep: number
): ActivationTensor {
  const output = createActivationTensor(input.channels, input.height, input.width);
  
  // Compute time embedding: t * weights + bias
  const embedding: number[] = [];
  for (let i = 0; i < timeEmbed.outputDim; i++) {
    embedding.push(timestep * timeEmbed.weights[i] + timeEmbed.bias[i]);
  }
  
  // Add to each spatial position
  for (let c = 0; c < input.channels; c++) {
    const embVal = c < embedding.length ? embedding[c] : 0;
    for (let h = 0; h < input.height; h++) {
      for (let w = 0; w < input.width; w++) {
        output.data[c][h][w] = input.data[c][h][w] + embVal;
      }
    }
  }
  
  return output;
}

/**
 * Run a complete forward pass through the tiny U-Net
 * Returns all intermediate activations for visualization
 */
export function forwardPass(
  model: TinyUNet,
  noisyInput: ActivationTensor,
  timestep: number
): ForwardPassState {
  // Input conv: 1ch → 2ch
  let x = applyConv2d(noisyInput, model.inputConv);
  x = applySiLU(x);
  const afterInputConv = x;
  
  // Add time embedding
  x = addTimeEmbedding(x, model.timeEmbed, timestep);
  
  // Encoder conv: 2ch → 2ch
  x = applyConv2d(x, model.encoderConv);
  x = applySiLU(x);
  const afterEncoder = x; // Save for skip connection
  
  // Downsample: 2×2 → 1×1
  x = downsample2x(x);
  const afterDownsample = x;
  
  // Bottleneck conv: 2ch → 2ch
  x = applyConv2d(x, model.bottleneckConv);
  x = applySiLU(x);
  const afterBottleneck = x;
  
  // Upsample: 1×1 → 2×2
  x = upsample2x(x);
  const afterUpsample = x;
  
  // Skip connection: concatenate encoder output
  x = concat(x, afterEncoder);
  const afterSkipConcat = x;
  
  // Decoder conv: 4ch → 2ch
  x = applyConv2d(x, model.decoderConv);
  x = applySiLU(x);
  const afterDecoder = x;
  
  // Output conv: 2ch → 1ch (no activation - predict noise directly)
  const output = applyConv2d(x, model.outputConv);
  
  return {
    noisyInput,
    timestep,
    afterInputConv,
    afterEncoder,
    afterDownsample,
    afterBottleneck,
    afterUpsample,
    afterSkipConcat,
    afterDecoder,
    output,
  };
}

/**
 * Create a sample noisy input for testing
 */
export function createSampleInput(): ActivationTensor {
  const input = createActivationTensor(1, 2, 2);
  // Simple pattern: gradient
  input.data[0][0][0] = 0.1;
  input.data[0][0][1] = 0.4;
  input.data[0][1][0] = 0.6;
  input.data[0][1][1] = 0.9;
  return input;
}

/**
 * Create random noisy input
 */
export function createRandomInput(): ActivationTensor {
  const input = createActivationTensor(1, 2, 2);
  for (let h = 0; h < 2; h++) {
    for (let w = 0; w < 2; w++) {
      input.data[0][h][w] = Math.random();
    }
  }
  return input;
}
