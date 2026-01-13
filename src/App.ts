import * as THREE from 'three';
import { SceneManager } from "./core/SceneManager";
import { createTinyUNet, forwardPass, createSampleInput, createRandomInput, countParameters } from "./models/TinyModelGenerator";
import { createTinyUNetWithActivations } from "./visualizations/TinyModelVis";
import { DEFAULT_TINY_VIS_CONFIG, TinyUNet, ForwardPassState } from "./types/tiny";
import { createInfoPanel } from "./components/controls/InfoPanel";

/**
 * Simplified application for DIAMOND architecture visualization
 * Shows the Tiny U-Net with all weights visible and forward pass activations
 */
export class Application {
  private sceneManager: SceneManager;
  private model: TinyUNet;
  private forwardState: ForwardPassState;
  private visualization: THREE.Group | null = null;
  private timestep: number = 0.5;

  constructor() {
    // Initialize scene manager
    this.sceneManager = new SceneManager();

    // Create the tiny model
    this.model = createTinyUNet();
    console.log(`Tiny U-Net created with ${countParameters(this.model)} parameters`);

    // Run initial forward pass
    const input = createSampleInput();
    this.forwardState = forwardPass(this.model, input, this.timestep);

    // Create visualization with activations
    this.updateVisualization();
  }

  /**
   * Update the visualization with current model and forward state
   */
  private updateVisualization(): void {
    // Remove old visualization
    if (this.visualization) {
      this.sceneManager.getScene().remove(this.visualization);
    }

    // Create new visualization
    this.visualization = createTinyUNetWithActivations(
      this.model,
      this.forwardState,
      DEFAULT_TINY_VIS_CONFIG
    );
    this.sceneManager.getScene().add(this.visualization);
  }

  /**
   * Run a new forward pass with random input
   */
  public runForwardPass(timestep?: number): void {
    if (timestep !== undefined) {
      this.timestep = timestep;
    }
    const input = createRandomInput();
    this.forwardState = forwardPass(this.model, input, this.timestep);
    this.updateVisualization();
  }

  /**
   * Reinitialize the model with new random weights
   */
  public reinitializeModel(): void {
    this.model = createTinyUNet();
    const input = createRandomInput();
    this.forwardState = forwardPass(this.model, input, this.timestep);
    this.updateVisualization();
  }


  /**
   * Start the application
   */
  public start(): HTMLElement {
    // Start the animation loop
    this.sceneManager.startAnimationLoop();

    // Add cleanup handler
    window.addEventListener('beforeunload', () => this.dispose());

    // Create info panel with controls
    const infoPanel = createInfoPanel(this.model, this.timestep, {
      onNewInput: () => this.runForwardPass(),
      onNewWeights: () => this.reinitializeModel(),
      onTimestepChange: (t) => {
        this.timestep = t;
        this.runForwardPass(this.timestep);
      }
    });

    return infoPanel;
  }

  /**
   * Get the scene manager instance
   */
  public getSceneManager(): SceneManager {
    return this.sceneManager;
  }

  /**
   * Clean up resources
   */
  public dispose(): void {
    this.sceneManager.dispose();
  }
}
