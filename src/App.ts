import { SceneManager } from "./core/SceneManager";
import { createMinimalUNet } from "./models/DiamondModelGenerator";
import { DiamondVisualizationManager } from "./visualizations/DiamondVisualizationManager";
import { InteractableType } from "./core/InteractionManager";

/**
 * Simplified application for DIAMOND architecture visualization
 * Supports multiple visualization modes and interactive exploration
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

    // Set up interactions
    this.setupInteractions();

    // Set up keyboard controls
    this.setupKeyboardControls();
  }

  /**
   * Set up interaction handlers for clicking on blocks
   */
  private setupInteractions(): void {
    const interactionManager = this.sceneManager.getInteractionManager();

    // Handle left-click on DIAMOND blocks
    interactionManager.getLeftClickStream().subscribe((data) => {
      if (data.type === InteractableType.DIAMOND_BLOCK) {
        const blockName = data.object.userData.blockName;
        this.vizManager.handleBlockClick(blockName);

        // Update info panel
        this.updateInfoPanelMode();
      }
    });
  }

  /**
   * Set up keyboard controls for switching modes
   */
  private setupKeyboardControls(): void {
    window.addEventListener('keydown', (event) => {
      switch (event.key.toLowerCase()) {
        case 'm':
          // Toggle between modes
          this.vizManager.cycleMode();
          this.updateInfoPanelMode();
          break;
        case 'escape':
          // Return to architecture view
          this.vizManager.showArchitectureView();
          this.updateInfoPanelMode();
          break;
        case 'a':
          // Show animation
          this.vizManager.showAnimationView();
          this.updateInfoPanelMode();
          break;
      }
    });
  }

  /**
   * Update info panel to show current mode
   */
  private updateInfoPanelMode(): void {
    const modeDisplay = document.getElementById('current-mode');
    if (modeDisplay) {
      const mode = this.vizManager.getCurrentMode();
      modeDisplay.textContent = mode.toUpperCase();
    }
  }

  /**
   * Start the application
   */
  public start(): HTMLElement {
    // Start the animation loop with update callback
    this.sceneManager.startAnimationLoop();

    // Add update loop for animations
    const animate = () => {
      this.vizManager.update();
      requestAnimationFrame(animate);
    };
    animate();

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
      <div style="margin-bottom: 10px;">
        <strong style="color: #fff;">Mode:</strong>
        <span id="current-mode" style="color: #00ff88;">ARCHITECTURE</span>
      </div>
      <div style="margin-bottom: 10px; color: #888;">
        <strong style="color: #fff;">Architecture:</strong><br/>
        • 4×4 input (RGB)<br/>
        • Channels: 3→8→16→8→3<br/>
        • ~1,200 parameters<br/>
        • 1 encoder + bottleneck + 1 decoder
      </div>
      <div style="color: #888; border-top: 1px solid #444; padding-top: 10px; margin-top: 10px;">
        <strong style="color: #fff;">Mouse:</strong><br/>
        • Left click: Select block<br/>
        • Middle click: Rotate<br/>
        • Right click: Pan<br/>
        • Scroll: Zoom
      </div>
      <div style="color: #888; border-top: 1px solid #444; padding-top: 10px; margin-top: 10px;">
        <strong style="color: #fff;">Keyboard:</strong><br/>
        • <kbd style="background: #333; padding: 2px 6px; border-radius: 3px;">M</kbd> Cycle modes<br/>
        • <kbd style="background: #333; padding: 2px 6px; border-radius: 3px;">A</kbd> Animation<br/>
        • <kbd style="background: #333; padding: 2px 6px; border-radius: 3px;">ESC</kbd> Back to architecture
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
