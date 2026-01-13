import * as THREE from "three";
import WebGPURenderer from "three/src/renderers/webgpu/WebGPURenderer.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { VisualizationManager } from "../visualizations/VisualizationManager";
import { PanelType, SceneConfig, DEFAULT_SCENE_CONFIG } from "../types/scene";
import { PanelLabels } from "../components/PanelLabels";

import { InteractionManager } from "./InteractionManager";

/**
 * SceneManager class responsible for the core 3D scene setup and management
 */
export class SceneManager {
  // THREE.js components
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: WebGPURenderer;
  private controls: OrbitControls;
  private panels: Map<PanelType, THREE.Group> = new Map();
  
  // Application components
  private visualizationManager: VisualizationManager;
  private panelLabels: PanelLabels;
  private interactionManager: InteractionManager;
  
  // Configuration
  private config: SceneConfig;
  
  // Left-click state tracking
  private isLeftMouseDown: boolean = false;
  
  // Raycasting for interaction detection
  private raycaster = new THREE.Raycaster();
  private mouse = new THREE.Vector2();

  constructor(config: Partial<SceneConfig> = {}) {
    // Merge provided config with defaults
    this.config = { ...DEFAULT_SCENE_CONFIG, ...config };
    
    // Initialize THREE.js components
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.config.renderer.backgroundColor);
    
    this.camera = new THREE.PerspectiveCamera(
      this.config.camera.fov,
      window.innerWidth / window.innerHeight,
      this.config.camera.near,
      this.config.camera.far
    );
    this.camera.position.copy(this.config.camera.initialPosition);
    
    // Initialize renderer
    this.renderer = new WebGPURenderer({ 
      antialias: this.config.renderer.antialias,
      powerPreference: 'high-performance' 
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    document.body.appendChild(this.renderer.domElement);
    
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    // Using OrbitControls defaults:
    // - enableDamping: false
    // - Left click = Rotate, Middle = Dolly/Zoom, Right = Pan
    
    // Add lights
    this.addLights();
    
    // Handle window resize
    window.addEventListener("resize", this.onWindowResize.bind(this));
    
    // Initialize visualizations
    this.visualizationManager = new VisualizationManager(this);
    
    // Initialize panel labels
    this.panelLabels = new PanelLabels();
    this.scene.add(this.panelLabels.getLabelGroup());
    
    // Initialize interaction manager
    this.interactionManager = new InteractionManager(this.camera, this.scene);
    
    // Set the canvas element for cursor management
    this.interactionManager.setCanvasElement(this.renderer.domElement);
    
    // Add mouse event listeners for selection indicator
    this.renderer.domElement.addEventListener('mousedown', this.onMouseDown.bind(this));
    this.renderer.domElement.addEventListener('mousemove', this.onMouseMove.bind(this));
    this.renderer.domElement.addEventListener('mouseup', this.onMouseUp.bind(this));
    this.renderer.domElement.addEventListener('mouseleave', this.onMouseLeave.bind(this));
    this.renderer.domElement.addEventListener('contextmenu', this.onContextMenu.bind(this));
  }
  
  /**
   * Add lights to the scene
   */
  private addLights(): void {
    // const ambientLight = new THREE.AmbientLight(0xcccccc, 0.4);
    // this.scene.add(ambientLight);
    
    // const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    // directionalLight.position.set(1, 1, 1);
    // this.scene.add(directionalLight);
  }
  
  /**
   * Handle window resize
   */
  private onWindowResize(): void {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }
  
  /**
   * Handle mouse down events
   */
  private onMouseDown(event: MouseEvent): void {
    if (event.button === 0) { // Left mouse button
      // Check if clicking on an interactable object
      this.updateMousePosition(event.clientX, event.clientY);
      const intersects = this.raycaster.intersectObjects(this.findInteractableObjects());
      
      if (intersects.length > 0) {
        // Clicking on an object - disable panning and start selection interaction
        this.isLeftMouseDown = true;
        this.controls.enabled = false; // Disable orbit controls temporarily
        
        // Start hover detection during left-click hold
        this.interactionManager.handleLeftClickHover(event.clientX, event.clientY);
        event.preventDefault(); // Prevent any default behavior
      } else {
        // Clicking on empty space - clear selection and allow normal panning
        this.interactionManager.clearAll();
        // Let the controls handle panning normally (don't prevent default)
      }
    }
  }

  /**
   * Handle mouse move events
   */
  private onMouseMove(event: MouseEvent): void {
    if (this.isLeftMouseDown) {
      // We're in selection mode - update hover detection
      this.interactionManager.handleLeftClickHover(event.clientX, event.clientY);
    } else {
      // Not holding left mouse - check for hover to show pointer cursor
      this.updateMousePosition(event.clientX, event.clientY);
      const intersects = this.raycaster.intersectObjects(this.findInteractableObjects());
      
      if (intersects.length > 0) {
        this.interactionManager.setPointerCursor();
      } else {
        this.interactionManager.setDefaultCursor();
      }
    }
  }

  /**
   * Handle mouse up events
   */
  private onMouseUp(event: MouseEvent): void {
    if (event.button === 0 && this.isLeftMouseDown) { // Left mouse button
      this.isLeftMouseDown = false;
      
      // Re-enable orbit controls
      this.controls.enabled = true;
      
      // Finalize the selection interaction
      this.interactionManager.handleLeftClickRelease(event.clientX, event.clientY);
      
      // Reset cursor based on what's under the mouse
      this.updateMousePosition(event.clientX, event.clientY);
      const intersects = this.raycaster.intersectObjects(this.findInteractableObjects());
      if (intersects.length > 0) {
        this.interactionManager.setPointerCursor();
      } else {
        this.interactionManager.setDefaultCursor();
      }
    }
  }

  /**
   * Handle mouse leave events - cancel left-click interaction if mouse leaves canvas
   */
  private onMouseLeave(_event: MouseEvent): void {
    if (this.isLeftMouseDown) {
      this.isLeftMouseDown = false;
      
      // Re-enable orbit controls
      this.controls.enabled = true;
      
      // Clear any hover state without finalizing selection
      this.interactionManager.clearLeftClickHover();
      
      // Reset cursor to default when leaving canvas
      this.interactionManager.setDefaultCursor();
    }
  }

  /**
   * Handle context menu events to prevent default behavior
   */
  private onContextMenu(event: MouseEvent): void {
    event.preventDefault(); // Prevent context menu
  }
  
  /**
   * Start the animation loop
   */
  public startAnimationLoop(): void {
    const animate = (): void => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    this.renderer.setAnimationLoop(animate);
  }
  
  /**
   * Clean up resources
   */
  public dispose(): void {
    this.visualizationManager.dispose();
    this.panelLabels.dispose();
    this.interactionManager.dispose();
    this.renderer.setAnimationLoop(null);
    window.removeEventListener("resize", this.onWindowResize.bind(this));
    this.renderer.domElement.removeEventListener('mousedown', this.onMouseDown.bind(this));
    this.renderer.domElement.removeEventListener('mousemove', this.onMouseMove.bind(this));
    this.renderer.domElement.removeEventListener('mouseup', this.onMouseUp.bind(this));
    this.renderer.domElement.removeEventListener('mouseleave', this.onMouseLeave.bind(this));
    this.renderer.domElement.removeEventListener('contextmenu', this.onContextMenu.bind(this));
  }
  
  /**
   * Update mouse position for raycasting
   */
  private updateMousePosition(screenX: number, screenY: number): void {
    this.mouse.x = (screenX / window.innerWidth) * 2 - 1;
    this.mouse.y = -(screenY / window.innerHeight) * 2 + 1;
    this.raycaster.setFromCamera(this.mouse, this.camera);
  }
  
  /**
   * Find all interactable objects in the scene
   */
  private findInteractableObjects(): THREE.Object3D[] {
    return this.interactionManager.findInteractableObjects();
  }

  // Panel management methods
  
  /**
   * Add a panel to the scene
   */
  public addPanel(name: PanelType, group: THREE.Group): void {
    // Define the proper panel order
    const panelOrder = [
      PanelType.TRAINING_DATA,
      PanelType.NEURAL_NETWORK,
      PanelType.PREDICTIONS,
      PanelType.LINES,
      PanelType.ANALYTICAL_POLYTOPES,
      PanelType.POLYTOPES
    ];

    // Calculate position based on the proper panel order
    let currentDepth = 0;
    const currentPanelIndex = panelOrder.indexOf(name);
    
    // Calculate position based on panels that should come before this one
    for (let i = 0; i < currentPanelIndex; i++) {
      const prevPanelType = panelOrder[i];
      const prevPanel = this.panels.get(prevPanelType);
      
      if (prevPanel) {
        const boundingBox = new THREE.Box3().setFromObject(prevPanel);
        const depth = boundingBox.max.z - boundingBox.min.z;
        currentDepth = prevPanel.position.z - depth - this.config.panelSpacing;
      }
    }
    
    group.position.z = currentDepth;
    this.panels.set(name, group);
    this.scene.add(group);
    
    // Add label for this panel with the panel group for Z-depth calculation
    this.panelLabels.setLabel(name, group.position, group);
  }
  
  /**
   * Update an existing panel
   */
  public updatePanel(name: PanelType, newGroup: THREE.Group): void {
    const oldGroup = this.panels.get(name);
    
    if (oldGroup) {
      newGroup.position.copy(oldGroup.position);
      this.scene.remove(oldGroup);
      this.scene.add(newGroup);
      this.panels.set(name, newGroup);
      
      // Update label position with the new panel group for Z-depth calculation
      this.panelLabels.updateLabelPosition(name, newGroup.position, newGroup);
      
      // Reposition panels that come after this one
      this.repositionPanelsAfter(name);
    } else {
      this.addPanel(name, newGroup);
    }
  }

  /**
   * Reposition all panels that come after the specified panel
   */
  private repositionPanelsAfter(changedPanelName: PanelType): void {
    const panelOrder = [
      PanelType.TRAINING_DATA,
      PanelType.NEURAL_NETWORK,
      PanelType.PREDICTIONS,
      PanelType.LINES,
      PanelType.ANALYTICAL_POLYTOPES,
      PanelType.POLYTOPES
    ];

    const changedPanelIndex = panelOrder.indexOf(changedPanelName);
    if (changedPanelIndex === -1 || changedPanelIndex === panelOrder.length - 1) {
      return; // No panels after this one
    }

    // Calculate new positions for panels that come after the changed panel
    let currentDepth = 0;
    
    for (let i = 0; i <= changedPanelIndex; i++) {
      const panelType = panelOrder[i];
      const panel = this.panels.get(panelType);
      
      if (panel) {
        if (i === 0) {
          // First panel stays at z=0
          currentDepth = 0;
        } else {
          // Calculate position based on previous panel
          const prevPanelType = panelOrder[i - 1];
          const prevPanel = this.panels.get(prevPanelType);
          
          if (prevPanel) {
            const boundingBox = new THREE.Box3().setFromObject(prevPanel);
            const depth = boundingBox.max.z - boundingBox.min.z;
            currentDepth = prevPanel.position.z - depth - this.config.panelSpacing;
          }
        }
        
        panel.position.z = currentDepth;
        // Update label position with panel group for Z-depth calculation
        this.panelLabels.updateLabelPosition(panelType, panel.position, panel);
      }
    }

    // Now reposition the panels that come after the changed panel
    for (let i = changedPanelIndex + 1; i < panelOrder.length; i++) {
      const panelType = panelOrder[i];
      const panel = this.panels.get(panelType);
      
      if (panel) {
        // Calculate position based on previous panel
        const prevPanelType = panelOrder[i - 1];
        const prevPanel = this.panels.get(prevPanelType);
        
        if (prevPanel) {
          const boundingBox = new THREE.Box3().setFromObject(prevPanel);
          const depth = boundingBox.max.z - boundingBox.min.z;
          currentDepth = prevPanel.position.z - depth - this.config.panelSpacing;
          panel.position.z = currentDepth;
          // Update label position with panel group for Z-depth calculation
          this.panelLabels.updateLabelPosition(panelType, panel.position, panel);
        }
      }
    }
  }
  
  /**
   * Toggle panel visibility
   */
  public togglePanelVisibility(name: PanelType, visible: boolean): void {
    const group = this.panels.get(name);
    if (group) {
      group.visible = visible;
    }
    
    // Also toggle label visibility
    this.panelLabels.setLabelVisibility(name, visible);
  }
  
  /**
   * Show or hide a panel
   */
  public setPanelVisibility(name: PanelType, visible: boolean): void {
    const panel = this.panels.get(name);
    if (panel) {
      panel.visible = visible;
      this.panelLabels.setLabelVisibility(name, visible);
    }
  }

  /**
   * Update panel content and visibility
   */
  public updatePanelWithVisibility(name: PanelType, newGroup: THREE.Group, visible: boolean): void {
    if (visible) {
      this.updatePanel(name, newGroup);
      this.setPanelVisibility(name, true);
    } else {
      this.setPanelVisibility(name, false);
    }
  }
  
  /**
   * Reset camera to default position
   */
  public resetCamera(): void {
    this.camera.position.copy(this.config.camera.initialPosition);
    this.camera.lookAt(0, 0, 0);
    this.controls.reset();
  }
  
  // Accessor methods
  
  public getScene(): THREE.Scene {
    return this.scene;
  }
  
  public getCamera(): THREE.PerspectiveCamera {
    return this.camera;
  }
  
  public getInteractionManager(): InteractionManager {
    return this.interactionManager;
  }
}