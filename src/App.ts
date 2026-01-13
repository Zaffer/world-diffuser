import { SceneManager } from "./core/SceneManager";
import { createMinimalUNet } from "./models/DiamondModelGenerator";
import { DiamondVisualizationManager } from "./visualizations/DiamondVisualizationManager";

/**
 * Simplified application for DIAMOND architecture visualization
 * Shows the U-Net with all weights visible inline
 */
export class Application {
  private sceneManager: SceneManager;
  private vizManager: DiamondVisualizationManager;

  constructor() {
    // Initialize scene manager
    this.sceneManager = new SceneManager();

    // Create the U-Net model
    const unetModel = createMinimalUNet();

    // Initialize visualization manager
    this.vizManager = new DiamondVisualizationManager(
      this.sceneManager.getScene(),
      unetModel
    );
  }


  /**
   * Start the application
   */
  public start(): HTMLElement {
    // Start the animation loop
    this.sceneManager.startAnimationLoop();

    // Add cleanup handler
    window.addEventListener('beforeunload', () => this.dispose());

    // Create simple info panel
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
      background: rgba(26, 26, 26, 0.9);
      border: 1px solid #444;
      border-radius: 8px;
      padding: 20px;
      color: #00ffff;
      font-family: monospace;
      font-size: 14px;
      z-index: 1000;
      max-width: 300px;
    `;

    panel.innerHTML = `
      <h2 style="margin: 0 0 15px 0; font-size: 18px; color: #00ffff;">DIAMOND U-Net</h2>
      <div style="margin-bottom: 10px; color: #888;">
        <strong style="color: #fff;">Architecture:</strong><br/>
        • 4×4 input (RGB)<br/>
        • Channels: 3→8→16→8→3<br/>
        • ~1,200 parameters<br/>
        • 1 encoder + bottleneck + 1 decoder
      </div>
      <div style="color: #888; border-top: 1px solid #444; padding-top: 10px; margin-top: 10px;">
        <strong style="color: #fff;">Controls:</strong><br/>
        • Middle click: Rotate<br/>
        • Right click: Pan<br/>
        • Scroll: Zoom
      </div>
      <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #444;">
        <strong style="color: #0f0;">🟢 Green</strong> = Positive weights<br/>
        <strong style="color: #f00;">🔴 Red</strong> = Negative weights<br/>
        <strong style="color: #888;">⚫ Gray</strong> = Near-zero
      </div>
    `;

    return panel;
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
