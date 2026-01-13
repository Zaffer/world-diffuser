import { SceneManager } from "./core/SceneManager";
import { createMinimalUNet } from "./models/DiamondModelGenerator";
import { createMinimalUNetVisualization } from "./visualizations/DiamondArchitectureVis";
import { DEFAULT_ARCH_VIS_CONFIG } from "./types/diamond";

/**
 * Simplified application for DIAMOND architecture visualization
 * Directly shows the U-Net architecture without panels or training logic
 */
export class Application {
  private sceneManager: SceneManager;

  constructor() {
    // Initialize scene manager
    this.sceneManager = new SceneManager();

    // Create and add the DIAMOND U-Net visualization directly
    const unetModel = createMinimalUNet();
    const architectureVis = createMinimalUNetVisualization(unetModel, DEFAULT_ARCH_VIS_CONFIG);
    this.sceneManager.scene.add(architectureVis);
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
