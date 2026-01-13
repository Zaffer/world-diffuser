Claude’s Plan
DIAMOND Minimal Architecture Visualizer - Implementation Plan
Overview
Create a 3D mechanistic visualizer for a minimal DIAMOND diffusion world model. Start with the smallest possible U-Net that demonstrates the core architecture, with every parameter visible and adjustable.

User Requirements
Layout: Vertical stack (encoder top, bottleneck center, decoder bottom)
Initial view: Full architecture overview showing all blocks as labeled components
Scale: Minimal model with very few parameters (just enough for basic functionality)
Detail: Every weight, bias, and parameter should be mechanistically visible
Goal: Understand the architecture itself, not illustrative visualizations
Minimal DIAMOND Architecture Specification
Tiny U-Net Configuration

Input: 4x4 RGB images (3 channels)
Channels: [8, 16] progression (minimal for demonstration)
Blocks: 1 encoder stage, 1 bottleneck, 1 decoder stage

Total structure:
- EncoderStage[0]: ResBlock (3→8 channels) + Downsample (4x4→2x2)
- Bottleneck: ResBlock (8→16 channels, 2x2 spatial)
- DecoderStage[0]: Upsample (2x2→4x4) + ResBlock (16→8 channels) + Skip from encoder
- Output: Conv2D (8→3 channels)

Total parameters: ~1,200 weights (all visible!)
Why This Size?
4x4 input: Small enough to visualize all activations
8/16 channels: Minimal to show channel progression
3x3 kernels: Standard conv size, 9 weights per kernel
1 stage each: Demonstrates encoder→bottleneck→decoder flow
Every weight visible: At 8 output × 3 input × 3×3 kernel = 216 weights for first conv, all can be displayed as cubes
Component Naming Convention
Block Naming
InputConv: Initial conv to project 3 channels → 8 channels
Encoder_Block0: First residual block (8→8 channels)
Encoder_Downsample: Spatial reduction 4x4→2x2
Bottleneck_Block: Center processing (8→16 channels)
Decoder_Upsample: Spatial expansion 2x2→4x4
Decoder_Block0: Final residual block (16→8 channels)
SkipConnection_E0_D0: Encoder block 0 → Decoder block 0
OutputConv: Final projection 8 channels → 3 channels
Layer Naming Within Blocks
Conv2D_1: First convolution in block
GroupNorm_1: First normalization
Activation_1_SiLU: First activation
Conv2D_2: Second convolution
ResidualPath: Skip connection within block
Parameter Naming
Weight: W[oc, ic, kh, kw] - output channel, input channel, kernel height/width
Bias: B[oc] - output channel
Norm scale: γ[c] - channel
Norm shift: β[c] - channel
3D Spatial Layout
Coordinate System (Vertical Stack)

Y-axis (Vertical):
  +6 units: OutputConv + labels
  +4 units: Decoder_Block0
  +2 units: Decoder_Upsample
   0 units: Bottleneck_Block (center, origin)
  -2 units: Encoder_Downsample
  -4 units: Encoder_Block0
  -6 units: InputConv

X-axis (Horizontal):
  Used for laying out channels/kernels within each block
  Range: ±3 units (symmetric around origin)

Z-axis (Depth):
  -2 units: Layer components within blocks
  Used for stacking layers within a block (Conv → Norm → Activation)
Block Visual Design
Each block rendered as:

Labeled container box (wireframe cube showing block bounds)
Conv layer kernels (grid of colored weight cubes)
Norm parameters (γ, β displayed as bars)
Activation function (symbolic representation)
Info label (block name, shape: [C, H, W])
Skip Connection Visualization
Curved purple line from Encoder_Block0 output to Decoder_Block0 input
Label: Shows tensor shape being passed: [8, 4, 4]
Thickness: Proportional to information (fixed for now)
Type System Updates
New Types to Add to src/types/diamond.ts

// Architecture hierarchy
export interface UNetStage {
  stageName: string;
  blocks: ResidualBlock[];
  downsample?: DownsampleLayer;
  upsample?: UpsampleLayer;
  spatialShape: [number, number]; // [H, W]
  channelCount: number;
}

export interface DownsampleLayer {
  type: LayerType.DOWNSAMPLE;
  method: 'conv_stride' | 'maxpool' | 'avgpool';
  factor: number; // 2 for 2x downsampling
  conv?: ConvLayer; // if method is conv_stride
}

export interface UpsampleLayer {
  type: LayerType.UPSAMPLE;
  method: 'conv_transpose' | 'nearest' | 'bilinear';
  factor: number;
  conv?: ConvLayer;
}

export interface SkipConnection {
  id: string;
  fromBlockName: string;
  toBlockName: string;
  fromShape: [number, number, number]; // [C, H, W]
  toShape: [number, number, number];
}

export interface MinimalUNet {
  inputConv: ConvLayer;
  encoder: UNetStage;
  bottleneck: ResidualBlock;
  decoder: UNetStage;
  skipConnections: SkipConnection[];
  outputConv: ConvLayer;
}
Visualization Config Extension

export interface ArchitectureVisConfig extends LayerVisualizationConfig {
  showBlockBounds: boolean;
  showSkipConnections: boolean;
  showTensorShapes: boolean;
  showParameterCounts: boolean;
  blockSpacing: number; // Y-axis spacing between blocks
  kernelDisplayLimit: number; // Max kernels to show per layer
}
Implementation Steps
Phase 1: Model Generation
File: src/models/DiamondModelGenerator.ts (extend existing)

Add createMinimalUNet() function
Generate: InputConv, EncoderStage, Bottleneck, DecoderStage, OutputConv
Initialize all weights with Xavier initialization (already implemented)
Keep track of spatial shapes at each stage: 4x4 → 2x2 → 4x4
Phase 2: Block Visualization Components
File: src/visualizations/DiamondVis.ts (extend existing)

createBlockContainer(blockName, position, bounds)

Wireframe box with label
createResidualBlockVis(residualBlock, position, config)

Stack: Conv1 → Norm1 → Activation1 → Conv2 → Norm2
Show residual path as curved line
createDownsampleVis(downsample, position)

Show arrow indicating spatial reduction
Label: "4x4 → 2x2"
createUpsampleVis(upsample, position)

Show arrow indicating spatial expansion
Label: "2x2 → 4x4"
createSkipConnectionVis(skipConnection, fromPos, toPos)

Curved purple line with arrowhead
Label showing tensor shape
Phase 3: Full Architecture Layout
File: src/visualizations/DiamondArchitectureVis.ts (new)

Main function: createMinimalUNetVisualization(model: MinimalUNet, config)
Layout blocks vertically according to coordinate system above
Connect skip connections
Add stage labels ("Encoder", "Bottleneck", "Decoder")
Return complete THREE.Group
Phase 4: Integration
File: src/visualizations/NetworkVis.ts (modify existing)

Replace current DIAMOND visualization with full architecture view
Remove temporary single-layer visualization
Call createMinimalUNetVisualization() instead
Phase 5: Interactive Controls (Future)
File: src/components/controls/ControlPanel.ts (extend later)

Add sliders for model size (currently fixed to minimal)
Add checkboxes: Show weights, Show skip connections, Show tensor shapes
Add button: "Zoom to block" for detailed inspection
Critical Files to Modify
src/types/diamond.ts - Add UNetStage, MinimalUNet, skip connection types
src/models/DiamondModelGenerator.ts - Add createMinimalUNet()
src/visualizations/DiamondVis.ts - Add block-level visualizations
src/visualizations/DiamondArchitectureVis.ts (NEW) - Full architecture layout
src/visualizations/NetworkVis.ts - Replace with architecture view
Verification Plan
Visual Verification
Start dev server: npm start
Open http://localhost:3000/
Check that visualization shows:
✓ All blocks vertically stacked (encoder, bottleneck, decoder)
✓ Block labels clearly visible
✓ Skip connection from encoder to decoder
✓ Tensor shapes displayed at each stage
✓ Individual conv kernels visible when zooming in
Parameter Count Verification

// In browser console:
const model = createMinimalUNet();
console.log('Total parameters:', countParameters(model));
// Expected: ~1,200 (manageable for full visualization)
Interaction Verification
Click on a block → should see detailed conv kernels
Hover over weight cubes → should show exact value
Camera controls should allow orbiting around the full architecture
All block names should be readable
Next Steps After This Plan
Once minimal architecture is working:

Add timestep embedding visualization
Add action conditioning (adaptive group norm)
Add data flow animation (show activations flowing through)
Add gradient visualization
Add controls to adjust channel counts
Add attention mechanisms (if used)
Scale up to more realistic sizes (64x64 images, more stages)
Design Principles
Mechanistic First: Show actual parameters, not abstractions
Progressive Disclosure: Start with overview, allow drilling down
Every Parameter Matters: No hidden weights, everything inspectable
Minimal Viable: Smallest model that demonstrates the concept
Extensible: Easy to add more stages/channels later
Success Criteria
✓ Can see the complete U-Net architecture in one view
✓ Can identify encoder, bottleneck, decoder, skip connections
✓ Can zoom into any block and see conv kernel weights as colored cubes
✓ Can understand data flow: 4x4 input → 2x2 bottleneck → 4x4 output
✓ Every parameter (all ~1,200 weights) is theoretically accessible
✓ Clear, consistent naming throughout the visualization
✓ No placeholder or abstract representations - all mechanistic