import * as THREE from "three";
import { Observable, Subject } from "rxjs";

/**
 * Types of neural network objects that can be interacted with
 */
export enum InteractableType {
  NETWORK_NODE = "network_neuron",  // Neural network neurons
  NETWORK_EDGE = "network_synapse", // Neural network synapses (connections)
  DIAMOND_BLOCK = "diamond_block",   // DIAMOND architecture blocks (clickable)
  DIAMOND_OPERATION = "diamond_operation" // Operations like upsample/downsample
}

/**
 * Interaction data for clicked neural network objects
 */
export interface InteractionData {
  type: InteractableType;
  object: THREE.Object3D;
  layerIndex: number;
  nodeIndex?: number;        // For neurons: index within the layer
  sourceNodeIndex?: number;  // For synapses: source neuron index
  targetNodeIndex?: number;  // For synapses: target neuron index
  position: THREE.Vector3;
}

/**
 * Manages 3D object interactions in the scene
 */
export class InteractionManager {
  private camera: THREE.Camera;
  private scene: THREE.Scene;
  private raycaster = new THREE.Raycaster();
  private mouse = new THREE.Vector2();
  
  // Canvas element for cursor management
  private canvasElement: HTMLCanvasElement | null = null;
  
  // Observable streams for interactions
  private rightClickSubject = new Subject<InteractionData>();
  private leftClickSubject = new Subject<InteractionData>();
  
  // Visual feedback for objects - Simple opacity approach
  private hoveredObject: THREE.Object3D | null = null;
  private selectedObject: THREE.Object3D | null = null;
  
  // Store original opacity values to restore later
  private originalOpacity = new Map<THREE.Object3D, number>();
  
  constructor(camera: THREE.Camera, scene: THREE.Scene) {
    this.camera = camera;
    this.scene = scene;
  }
  
  /**
   * Set the canvas element for cursor management
   */
  public setCanvasElement(canvas: HTMLCanvasElement): void {
    this.canvasElement = canvas;
  }
  
  /**
   * Set cursor to crosshair for right-click interaction
   */
  public setCrosshairCursor(): void {
    if (this.canvasElement) {
      this.canvasElement.style.cursor = 'crosshair';
    }
  }
  
  /**
   * Set cursor to pointer when hovering over selectable object
   */
  public setPointerCursor(): void {
    if (this.canvasElement) {
      this.canvasElement.style.cursor = 'pointer';
    }
  }
  
  /**
   * Reset cursor to default
   */
  public setDefaultCursor(): void {
    if (this.canvasElement) {
      this.canvasElement.style.cursor = 'default';
    }
  }
  
  /**
   * Handle hover during left-click hold
   */
  public handleLeftClickHover(screenX: number, screenY: number): void {
    // Use simple raycasting to find object under cursor
    this.updateMousePosition(screenX, screenY);
    const intersects = this.raycaster.intersectObjects(this.findInteractableObjects());
    const newHoveredObject = intersects.length > 0 ? intersects[0].object : null;
    
    // Update hover state if changed
    if (newHoveredObject !== this.hoveredObject) {
      this.clearHoverHighlight();
      
      if (newHoveredObject) {
        this.hoveredObject = newHoveredObject;
        this.applyHoverHighlight(this.hoveredObject);
        // Change cursor to 'pointer' when hovering over selectable object
        this.setPointerCursor();
      } else {
        // Change cursor back to 'default' when not hovering over any object
        this.setDefaultCursor();
      }
    }
  }

  /**
   * Handle left-click release - finalize selection
   */
  public handleLeftClickRelease(screenX: number, screenY: number): void {
    this.updateMousePosition(screenX, screenY);
    
    // Clear any previous selection
    this.clearSelectionHighlight();
    
    // If we have a hovered object, make it the selected object
    if (this.hoveredObject) {
      const userData = this.hoveredObject.userData;
      
      // Create interaction data
      const interactionData: InteractionData = {
        type: userData.type,
        object: this.hoveredObject,
        layerIndex: userData.layerIndex,
        nodeIndex: userData.nodeIndex,
        sourceNodeIndex: userData.sourceNodeIndex,
        targetNodeIndex: userData.targetNodeIndex,
        position: new THREE.Vector3() // Will be updated by raycaster if needed
      };
      
      // Move hovered object to selected state
      this.selectedObject = this.hoveredObject;
      this.hoveredObject = null;
      
      // Apply selection highlight (different from hover)
      this.applySelectionHighlight(this.selectedObject);
      
      // Emit the interaction
      this.leftClickSubject.next(interactionData);
    } else {
      // Clear hover if no object was under cursor
      this.clearHoverHighlight();
    }
  }

  /**
   * Clear left-click hover state
   */
  public clearLeftClickHover(): void {
    this.clearHoverHighlight();
    // Reset cursor to default when clearing hover
    this.setDefaultCursor();
  }

  /**
   * Get observable for left-click interactions
   */
  public getLeftClickStream(): Observable<InteractionData> {
    return this.leftClickSubject.asObservable();
  }

  /**
   * Handle hover during right-click hold
   */
  public handleRightClickHover(screenX: number, screenY: number): void {
    // Use simple raycasting to find object under cursor
    this.updateMousePosition(screenX, screenY);
    const intersects = this.raycaster.intersectObjects(this.findInteractableObjects());
    const newHoveredObject = intersects.length > 0 ? intersects[0].object : null;
    
    // Update hover state if changed
    if (newHoveredObject !== this.hoveredObject) {
      this.clearHoverHighlight();
      
      if (newHoveredObject) {
        this.hoveredObject = newHoveredObject;
        this.applyHoverHighlight(this.hoveredObject);
        // Change cursor to 'pointer' when hovering over selectable object
        this.setPointerCursor();
      } else {
        // Change cursor back to 'crosshair' when not hovering over any object
        this.setCrosshairCursor();
      }
    }
  }
  
  /**
   * Handle right-click release - finalize selection
   */
  public handleRightClickRelease(screenX: number, screenY: number): void {
    this.updateMousePosition(screenX, screenY);
    
    // Clear any previous selection
    this.clearSelectionHighlight();
    
    // If we have a hovered object, make it the selected object
    if (this.hoveredObject) {
      const userData = this.hoveredObject.userData;
      
      // Create interaction data
      const interactionData: InteractionData = {
        type: userData.type,
        object: this.hoveredObject,
        layerIndex: userData.layerIndex,
        nodeIndex: userData.nodeIndex,
        sourceNodeIndex: userData.sourceNodeIndex,
        targetNodeIndex: userData.targetNodeIndex,
        position: new THREE.Vector3() // Will be updated by raycaster if needed
      };
      
      // Move hovered object to selected state
      this.selectedObject = this.hoveredObject;
      this.hoveredObject = null;
      
      // Apply selection highlight (different from hover)
      this.applySelectionHighlight(this.selectedObject);
      
      // Emit the interaction
      this.rightClickSubject.next(interactionData);
    } else {
      // Clear hover if no object was under cursor
      this.clearHoverHighlight();
    }
  }
  
  /**
   * Clear right-click hover state
   */
  public clearRightClickHover(): void {
    this.clearHoverHighlight();
    // Reset cursor to crosshair when clearing hover (still in right-click mode)
    this.setCrosshairCursor();
  }
  
  /**
   * Get observable for right-click interactions
   */
  public getRightClickStream(): Observable<InteractionData> {
    return this.rightClickSubject.asObservable();
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
  public findInteractableObjects(): THREE.Object3D[] {
    const interactables: THREE.Object3D[] = [];

    this.scene.traverse((child) => {
      if (child.userData.type === InteractableType.NETWORK_NODE ||
          child.userData.type === InteractableType.NETWORK_EDGE ||
          child.userData.type === InteractableType.DIAMOND_BLOCK ||
          child.userData.type === InteractableType.DIAMOND_OPERATION) {
        interactables.push(child);
      }
    });

    return interactables;
  }

  /**
   * Apply hover highlight using opacity change
   */
  private applyHoverHighlight(object: THREE.Object3D): void {
    const mesh = object as THREE.Mesh;
    const material = mesh.material as THREE.Material;
    
    // Store original opacity if not already stored
    if (!this.originalOpacity.has(object)) {
      this.originalOpacity.set(object, material.opacity);
    }
    
    // Make slightly more opaque on hover (0.9)
    material.opacity = 0.9;
    material.transparent = true;
  }
  
  /**
   * Apply selection highlight using opacity change
   */
  private applySelectionHighlight(object: THREE.Object3D): void {
    const mesh = object as THREE.Mesh;
    const material = mesh.material as THREE.Material;
    
    // Store original opacity if not already stored
    if (!this.originalOpacity.has(object)) {
      this.originalOpacity.set(object, material.opacity);
    }
    
    // Make completely opaque on selection (1.0)
    material.opacity = 1.0;
    material.transparent = false;
  }

  /**
   * Clear hover highlight
   */
  private clearHoverHighlight(): void {
    if (this.hoveredObject) {
      this.restoreOriginalOpacity(this.hoveredObject);
      this.hoveredObject = null;
    }
  }

  /**
   * Clear selection highlight
   */
  private clearSelectionHighlight(): void {
    if (this.selectedObject) {
      this.restoreOriginalOpacity(this.selectedObject);
      this.selectedObject = null;
    }
  }
  
  /**
   * Restore original opacity
   */
  private restoreOriginalOpacity(object: THREE.Object3D): void {
    const mesh = object as THREE.Mesh;
    const material = mesh.material as THREE.Material;
    const originalOpacity = this.originalOpacity.get(object);
    
    if (originalOpacity !== undefined) {
      material.opacity = originalOpacity;
      material.transparent = originalOpacity < 1.0;
    }
  }
  
  /**
   * Clear all highlights and selections
   */
  public clearAll(): void {
    this.clearHoverHighlight();
    this.clearSelectionHighlight();
  }

  /**
   * Clean up resources
   */
  public dispose(): void {
    this.clearAll();
    this.originalOpacity.clear();
    this.rightClickSubject.complete();
    this.leftClickSubject.complete();
  }

  /**
   * Update the selected object after NetworkVis regeneration
   * This finds the new object that matches the same metadata and updates the selection
   */
  public updateSelectedObjectAfterRegeneration(layerIndex: number, nodeIndex?: number, sourceNodeIndex?: number, targetNodeIndex?: number): THREE.Object3D | null {
    if (!this.selectedObject) return null;
    
    // Find the new object with matching metadata
    let newSelectedObject: THREE.Object3D | null = null;
    
    this.scene.traverse((child: THREE.Object3D) => {
      const userData = child.userData;
      
      // Check if this is the same logical object as before
      if (userData.layerIndex === layerIndex) {
        if (nodeIndex !== undefined && userData.nodeIndex === nodeIndex && userData.type === InteractableType.NETWORK_NODE) {
          newSelectedObject = child;
        } else if (sourceNodeIndex !== undefined && targetNodeIndex !== undefined && 
                   userData.sourceNodeIndex === sourceNodeIndex && 
                   userData.targetNodeIndex === targetNodeIndex && 
                   userData.type === InteractableType.NETWORK_EDGE) {
          newSelectedObject = child;
        }
      }
    });
    
    if (newSelectedObject) {
      // Clear the old selection without restoring opacity (since the old object is gone)
      this.selectedObject = null;
      
      // Set the new object as selected
      this.selectedObject = newSelectedObject;
      this.applySelectionHighlight(this.selectedObject);
    }
    
    return newSelectedObject;
  }
}
