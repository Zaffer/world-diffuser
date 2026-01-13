import * as THREE from 'three';
import { SceneManager } from "./core/SceneManager";
import { createTinyUNet, forwardPass, createSampleInput, createRandomInput, countParameters } from "./models/TinyModelGenerator";
import { createTinyUNetWithActivations } from "./visualizations/TinyModelVis";
import { DEFAULT_TINY_VIS_CONFIG, TinyUNet, ForwardPassState } from "./types/tiny";

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
    const infoPanel = this.createInfoPanel();

    return infoPanel;
  }

  /**
   * Create a simple info panel showing controls
   */
  private createInfoPanel(): HTMLElement {
    const panel = document.createElement('div');
    panel.style.cssText = `
      position: fixed;
      top: 20px;
      left: 20px;
      background: rgba(26, 26, 26, 0.95);
      border: 1px solid #444;
      border-radius: 8px;
      padding: 20px;
      color: #00ffff;
      font-family: monospace;
      font-size: 14px;
      z-index: 1000;
      max-width: 320px;
    `;

    const paramCount = countParameters(this.model);

    panel.innerHTML = `
      <h2 style="margin: 0 0 15px 0; font-size: 18px; color: #00ffff;">Tiny Diffusion U-Net</h2>
      <div style="margin-bottom: 10px; color: #888;">
        <strong style="color: #fff;">Architecture:</strong><br/>
        • 2×2 input (grayscale)<br/>
        • Channels: 1→2→2→2→1<br/>
        • <strong style="color: #0f0;">${paramCount} parameters</strong> (all visible!)<br/>
        • Each cube = 1 weight
      </div>
      <div style="margin-bottom: 10px; color: #888; border-top: 1px solid #444; padding-top: 10px;">
        <strong style="color: #fff;">Timestep:</strong> t = ${this.timestep.toFixed(2)}
      </div>
      <div style="margin-bottom: 10px;">
        <button id="btn-forward" style="background: #333; color: #0ff; border: 1px solid #0ff; padding: 8px 12px; margin: 2px; cursor: pointer; border-radius: 4px;">
          🔄 New Input
        </button>
        <button id="btn-reinit" style="background: #333; color: #f80; border: 1px solid #f80; padding: 8px 12px; margin: 2px; cursor: pointer; border-radius: 4px;">
          🎲 New Weights
        </button>
      </div>
      <div style="margin-bottom: 10px;">
        <label style="color: #fff;">Timestep t:</label><br/>
        <input id="slider-t" type="range" min="0" max="1" step="0.01" value="${this.timestep}" 
          style="width: 100%; margin-top: 5px;">
      </div>
      <div style="color: #888; border-top: 1px solid #444; padding-top: 10px; margin-top: 10px;">
        <strong style="color: #fff;">Controls:</strong><br/>
        • Left drag: Rotate<br/>
        • Right drag: Pan<br/>
        • Scroll: Zoom
      </div>
      <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #444;">
        <strong style="color: #4488ff;">🔵 Blue</strong> = Positive weights<br/>
        <strong style="color: #ff4444;">🔴 Red</strong> = Negative weights<br/>
        <strong style="color: #888;">Size</strong> = Weight magnitude
      </div>
      <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #444; color: #888;">
        <strong style="color: #fff;">Activations:</strong><br/>
        Colored planes below show data flowing through the network
      </div>
    `;

    // Add event listeners after panel is in DOM
    setTimeout(() => {
      document.getElementById('btn-forward')?.addEventListener('click', () => {
        this.runForwardPass();
        this.updateInfoPanel(panel);
      });

      document.getElementById('btn-reinit')?.addEventListener('click', () => {
        this.reinitializeModel();
        this.updateInfoPanel(panel);
      });

      document.getElementById('slider-t')?.addEventListener('input', (e) => {
        this.timestep = parseFloat((e.target as HTMLInputElement).value);
        this.runForwardPass(this.timestep);
        this.updateInfoPanel(panel);
      });
    }, 0);

    return panel;
  }

  /**
   * Update the timestep display in the info panel
   */
  private updateInfoPanel(panel: HTMLElement): void {
    const timestepDiv = panel.querySelector('div:nth-child(3)') as HTMLElement;
    if (timestepDiv) {
      timestepDiv.innerHTML = `<strong style="color: #fff;">Timestep:</strong> t = ${this.timestep.toFixed(2)}`;
    }
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
