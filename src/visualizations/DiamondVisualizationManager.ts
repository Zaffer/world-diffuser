/**
 * Manages different visualization modes for DIAMOND architecture
 * Handles switching between:
 * - Architecture view (high-level blocks)
 * - Detail view (weights and activations)
 * - Animation view (data flow)
 */

import * as THREE from 'three';
import { MinimalUNet, ResidualBlock, FeatureMap } from '../types/diamond';
import { createMinimalUNetVisualization } from './DiamondArchitectureVis';
import {
  createResidualBlockDetailView,
  createFeatureMapVisualization,
  createDataFlowParticles,
  updateDataFlowParticles,
} from './DetailedLayerVis';
import { DEFAULT_ARCH_VIS_CONFIG } from '../types/diamond';

export enum VisualizationMode {
  ARCHITECTURE = 'architecture',
  WEIGHTS = 'weights',
  ACTIVATIONS = 'activations',
  ANIMATION = 'animation',
}

/**
 * Manages the DIAMOND visualization state and mode transitions
 */
export class DiamondVisualizationManager {
  private scene: THREE.Scene;
  private model: MinimalUNet;
  private currentMode: VisualizationMode = VisualizationMode.ARCHITECTURE;
  private currentGroup: THREE.Group | null = null;

  // For animation mode
  private animationGroup: THREE.Group | null = null;
  private lastTime: number = 0;

  // Track selected block for detail view
  private selectedBlockName: string | null = null;

  constructor(scene: THREE.Scene, model: MinimalUNet) {
    this.scene = scene;
    this.model = model;

    // Start with architecture view
    this.showArchitectureView();
  }

  /**
   * Show high-level architecture view
   */
  public showArchitectureView(): void {
    this.clearCurrentView();
    this.currentMode = VisualizationMode.ARCHITECTURE;

    const architectureVis = createMinimalUNetVisualization(
      this.model,
      DEFAULT_ARCH_VIS_CONFIG
    );

    this.currentGroup = architectureVis;
    this.scene.add(architectureVis);
  }

  /**
   * Show detailed weight view for a specific block
   */
  public showWeightView(blockName: string): void {
    this.clearCurrentView();
    this.currentMode = VisualizationMode.WEIGHTS;
    this.selectedBlockName = blockName;

    // Find the appropriate block
    let block: ResidualBlock | null = null;

    if (blockName.includes('Encoder')) {
      block = this.model.encoder.blocks[0];
    } else if (blockName.includes('Bottleneck')) {
      block = this.model.bottleneck;
    } else if (blockName.includes('Decoder')) {
      block = this.model.decoder.blocks[0];
    }

    if (block) {
      const detailView = createResidualBlockDetailView(block);
      this.currentGroup = detailView;
      this.scene.add(detailView);
    }
  }

  /**
   * Show activation/feature map view
   */
  public showActivationView(featureMap: FeatureMap, channelIndex: number = 0): void {
    this.clearCurrentView();
    this.currentMode = VisualizationMode.ACTIVATIONS;

    const activationVis = createFeatureMapVisualization(featureMap, channelIndex);
    this.currentGroup = activationVis;
    this.scene.add(activationVis);
  }

  /**
   * Show animated data flow through the network
   */
  public showAnimationView(): void {
    this.clearCurrentView();
    this.currentMode = VisualizationMode.ANIMATION;

    // Create architecture as base
    const architectureVis = createMinimalUNetVisualization(
      this.model,
      DEFAULT_ARCH_VIS_CONFIG
    );
    this.currentGroup = architectureVis;
    this.scene.add(architectureVis);

    // Add animated particles
    this.animationGroup = new THREE.Group();

    // Define key positions for data flow
    const positions = {
      input: new THREE.Vector3(-9, 9, 0),
      encoder: new THREE.Vector3(-9, 0, 0),
      bottleneck: new THREE.Vector3(0, -9, 0),
      decoder: new THREE.Vector3(9, 0, 0),
      output: new THREE.Vector3(9, 9, 0),
    };

    // Create particle flows
    const flows = [
      { from: positions.input, to: positions.encoder },
      { from: positions.encoder, to: positions.bottleneck },
      { from: positions.bottleneck, to: positions.decoder },
      { from: positions.decoder, to: positions.output },
    ];

    flows.forEach((flow) => {
      const particles = createDataFlowParticles(flow.from, flow.to, 15);
      this.animationGroup!.add(particles);
    });

    this.scene.add(this.animationGroup);
    this.lastTime = performance.now();
  }

  /**
   * Handle click on a block - switch to detail view
   */
  public handleBlockClick(blockName: string): void {
    console.log(`Clicked on block: ${blockName}`);

    if (this.currentMode === VisualizationMode.ARCHITECTURE) {
      // Switch to weight view for this block
      this.showWeightView(blockName);
    } else {
      // Go back to architecture view
      this.showArchitectureView();
    }
  }

  /**
   * Update animation if in animation mode
   */
  public update(): void {
    if (this.currentMode === VisualizationMode.ANIMATION && this.animationGroup) {
      const now = performance.now();
      const deltaTime = (now - this.lastTime) / 1000; // Convert to seconds
      this.lastTime = now;

      // Update all particle groups
      this.animationGroup.children.forEach((child) => {
        if (child instanceof THREE.Group) {
          updateDataFlowParticles(child, deltaTime);
        }
      });
    }
  }

  /**
   * Get current visualization mode
   */
  public getCurrentMode(): VisualizationMode {
    return this.currentMode;
  }

  /**
   * Cycle to next visualization mode
   */
  public cycleMode(): void {
    const modes = [
      VisualizationMode.ARCHITECTURE,
      VisualizationMode.ANIMATION,
    ];

    const currentIndex = modes.indexOf(this.currentMode);
    const nextIndex = (currentIndex + 1) % modes.length;
    const nextMode = modes[nextIndex];

    switch (nextMode) {
      case VisualizationMode.ARCHITECTURE:
        this.showArchitectureView();
        break;
      case VisualizationMode.ANIMATION:
        this.showAnimationView();
        break;
    }
  }

  /**
   * Clear the current visualization
   */
  private clearCurrentView(): void {
    if (this.currentGroup) {
      this.scene.remove(this.currentGroup);
      this.currentGroup = null;
    }

    if (this.animationGroup) {
      this.scene.remove(this.animationGroup);
      this.animationGroup = null;
    }
  }

  /**
   * Clean up resources
   */
  public dispose(): void {
    this.clearCurrentView();
  }
}

/**
 * Generate a sample feature map for demonstration
 */
export function generateSampleFeatureMap(
  channels: number,
  height: number,
  width: number
): FeatureMap {
  const data: number[][][] = [];

  for (let c = 0; c < channels; c++) {
    const channelData: number[][] = [];
    for (let h = 0; h < height; h++) {
      const row: number[] = [];
      for (let w = 0; w < width; w++) {
        // Generate interesting patterns
        const x = (w / width) * 2 - 1;
        const y = (h / height) * 2 - 1;
        const r = Math.sqrt(x * x + y * y);
        const angle = Math.atan2(y, x);

        // Different patterns for different channels
        let value = 0;
        if (c === 0) {
          // Radial pattern
          value = Math.cos(r * Math.PI * 2) * 0.5 + 0.5;
        } else if (c === 1) {
          // Spiral pattern
          value = Math.sin(angle * 3 + r * 5) * 0.5 + 0.5;
        } else {
          // Checkerboard
          value = ((h + w) % 2) * 0.5 + 0.25;
        }

        row.push(value);
      }
      channelData.push(row);
    }
    data.push(channelData);
  }

  return {
    channels,
    height,
    width,
    data,
  };
}
